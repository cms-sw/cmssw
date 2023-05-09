#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "EventFilter/HcalRawToDigi/interface/HcalUHTRData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "PackerHelp.h"

#include <fstream>
#include <iostream>
#include <memory>

#include <sstream>
#include <string>

/* QUESTION: what about dual FED readout? */
/* QUESTION: what do I do if the number of 16-bit words
   are not divisible by 4? -- these need to
   fit into the 64-bit words of the FEDRawDataFormat */

using namespace std;

class HcalDigiToRawuHTR : public edm::global::EDProducer<> {
public:
  explicit HcalDigiToRawuHTR(const edm::ParameterSet&);
  ~HcalDigiToRawuHTR() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const int _verbosity;
  const vector<int> tdc1_;
  const vector<int> tdc2_;
  const bool packHBTDC_;
  static constexpr int tdcmax_ = 49;

  const edm::EDGetTokenT<HcalDataFrameContainer<QIE10DataFrame>> tok_QIE10DigiCollection_;
  const edm::EDGetTokenT<HcalDataFrameContainer<QIE11DataFrame>> tok_QIE11DigiCollection_;
  const edm::EDGetTokenT<HBHEDigiCollection> tok_HBHEDigiCollection_;
  const edm::EDGetTokenT<HFDigiCollection> tok_HFDigiCollection_;
  const edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_TPDigiCollection_;
  const edm::ESGetToken<HcalElectronicsMap, HcalElectronicsMapRcd> tok_electronicsMap_;

  const bool premix_;
};

HcalDigiToRawuHTR::HcalDigiToRawuHTR(const edm::ParameterSet& iConfig)
    : _verbosity(iConfig.getUntrackedParameter<int>("Verbosity", 0)),
      tdc1_(iConfig.getParameter<vector<int>>("tdc1")),
      tdc2_(iConfig.getParameter<vector<int>>("tdc2")),
      packHBTDC_(iConfig.getParameter<bool>("packHBTDC")),
      tok_QIE10DigiCollection_(
          consumes<HcalDataFrameContainer<QIE10DataFrame>>(iConfig.getParameter<edm::InputTag>("QIE10"))),
      tok_QIE11DigiCollection_(
          consumes<HcalDataFrameContainer<QIE11DataFrame>>(iConfig.getParameter<edm::InputTag>("QIE11"))),
      tok_HBHEDigiCollection_(consumes<HBHEDigiCollection>(iConfig.getParameter<edm::InputTag>("HBHEqie8"))),
      tok_HFDigiCollection_(consumes<HFDigiCollection>(iConfig.getParameter<edm::InputTag>("HFqie8"))),
      tok_TPDigiCollection_(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("TP"))),
      tok_electronicsMap_(esConsumes<HcalElectronicsMap, HcalElectronicsMapRcd>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("ElectronicsMap")))),
      premix_(iConfig.getParameter<bool>("premix")) {
  produces<FEDRawDataCollection>("");
  for (size_t i = 0; i < tdc1_.size(); i++) {
    if (!(tdc1_.at(i) >= 0 && tdc1_.at(i) <= tdc2_.at(i) && tdc2_.at(i) <= tdcmax_))
      edm::LogWarning("HcalDigiToRawuHTR")
          << " incorrect TDC ranges " << i << "-th element: " << tdc1_.at(i) << ", " << tdc2_.at(i) << ", " << tdcmax_;
  }
}

HcalDigiToRawuHTR::~HcalDigiToRawuHTR() {}

void HcalDigiToRawuHTR::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  edm::ESHandle<HcalElectronicsMap> item = iSetup.getHandle(tok_electronicsMap_);
  const HcalElectronicsMap* readoutMap = item.product();

  //collection to be inserted into event
  std::unique_ptr<FEDRawDataCollection> fed_buffers(new FEDRawDataCollection());

  //
  //  Extracting All the Collections containing useful Info
  edm::Handle<QIE10DigiCollection> qie10DigiCollection;
  edm::Handle<QIE11DigiCollection> qie11DigiCollection;
  edm::Handle<HBHEDigiCollection> hbheDigiCollection;
  edm::Handle<HFDigiCollection> hfDigiCollection;
  edm::Handle<HcalTrigPrimDigiCollection> tpDigiCollection;
  iEvent.getByToken(tok_QIE10DigiCollection_, qie10DigiCollection);
  iEvent.getByToken(tok_QIE11DigiCollection_, qie11DigiCollection);
  iEvent.getByToken(tok_HBHEDigiCollection_, hbheDigiCollection);
  iEvent.getByToken(tok_HFDigiCollection_, hfDigiCollection);
  iEvent.getByToken(tok_TPDigiCollection_, tpDigiCollection);

  // first argument is the fedid (minFEDID+crateId)
  map<int, unique_ptr<HCalFED>> fedMap;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // QIE10 precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  UHTRpacker uhtrs;
  // loop over each digi and allocate memory for each
  if (qie10DigiCollection.isValid()) {
    const QIE10DigiCollection& qie10dc = *(qie10DigiCollection);
    for (unsigned int j = 0; j < qie10dc.size(); j++) {
      QIE10DataFrame qiedf = static_cast<QIE10DataFrame>(qie10dc[j]);
      DetId detid = qiedf.detid();
      HcalElectronicsId eid(readoutMap->lookup(detid));
      int crateId = eid.crateId();
      int slotId = eid.slot();
      int uhtrIndex = ((slotId & 0xF) << 8) | (crateId & 0xFF);
      int presamples = qiedf.presamples();

      /* Defining a custom index that will encode only
	 the information about the crate and slot of a
	 given channel:   crate: bits 0-7
	 slot:  bits 8-12 */

      if (!uhtrs.exist(uhtrIndex)) {
        uhtrs.newUHTR(uhtrIndex, presamples);
      }
      uhtrs.addChannel(uhtrIndex, qiedf, readoutMap, _verbosity);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // QIE11 precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //UHTRpacker uhtrs;
  // loop over each digi and allocate memory for each
  if (qie11DigiCollection.isValid()) {
    const QIE11DigiCollection& qie11dc = *(qie11DigiCollection);
    for (unsigned int j = 0; j < qie11dc.size(); j++) {
      QIE11DataFrame qiedf = static_cast<QIE11DataFrame>(qie11dc[j]);
      DetId detid = qiedf.detid();
      HcalElectronicsId eid(readoutMap->lookup(detid));
      int crateId = eid.crateId();
      int slotId = eid.slot();
      int uhtrIndex = ((slotId & 0xF) << 8) | (crateId & 0xFF);
      int presamples = qiedf.presamples();

      //   convert to hb qie data if hb
      if (packHBTDC_ && HcalDetId(detid.rawId()).subdet() == HcalSubdetector::HcalBarrel)
        qiedf = convertHB(qiedf, tdc1_, tdc2_, tdcmax_);

      if (!uhtrs.exist(uhtrIndex)) {
        uhtrs.newUHTR(uhtrIndex, presamples);
      }
      uhtrs.addChannel(uhtrIndex, qiedf, readoutMap, _verbosity);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // HF (QIE8) precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // loop over each digi and allocate memory for each
  if (hfDigiCollection.isValid()) {
    const HFDigiCollection& qie8hfdc = *(hfDigiCollection);
    for (HFDigiCollection::const_iterator qiedf = qie8hfdc.begin(); qiedf != qie8hfdc.end(); qiedf++) {
      DetId detid = qiedf->id();

      HcalElectronicsId eid(readoutMap->lookup(detid));
      int crateId = eid.crateId();
      int slotId = eid.slot();
      int uhtrIndex = (crateId & 0xFF) | ((slotId & 0xF) << 8);
      int presamples = qiedf->presamples();

      if (!uhtrs.exist(uhtrIndex)) {
        uhtrs.newUHTR(uhtrIndex, presamples);
      }
      uhtrs.addChannel(uhtrIndex, qiedf, readoutMap, premix_, _verbosity);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // HBHE (QIE8) precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // loop over each digi and allocate memory for each
  if (hbheDigiCollection.isValid()) {
    const HBHEDigiCollection& qie8hbhedc = *(hbheDigiCollection);
    for (HBHEDigiCollection::const_iterator qiedf = qie8hbhedc.begin(); qiedf != qie8hbhedc.end(); qiedf++) {
      DetId detid = qiedf->id();

      HcalElectronicsId eid(readoutMap->lookup(detid));
      int crateId = eid.crateId();
      int slotId = eid.slot();
      int uhtrIndex = (crateId & 0xFF) | ((slotId & 0xF) << 8);
      int presamples = qiedf->presamples();

      if (!uhtrs.exist(uhtrIndex)) {
        uhtrs.newUHTR(uhtrIndex, presamples);
      }
      uhtrs.addChannel(uhtrIndex, qiedf, readoutMap, premix_, _verbosity);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // TP data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // loop over each digi and allocate memory for each
  if (tpDigiCollection.isValid()) {
    const HcalTrigPrimDigiCollection& qietpdc = *(tpDigiCollection);
    for (HcalTrigPrimDigiCollection::const_iterator qiedf = qietpdc.begin(); qiedf != qietpdc.end(); qiedf++) {
      DetId detid = qiedf->id();
      HcalElectronicsId eid(readoutMap->lookupTrigger(detid));

      int crateId = eid.crateId();
      int slotId = eid.slot();
      int uhtrIndex = (crateId & 0xFF) | ((slotId & 0xF) << 8);
      int ilink = eid.fiberIndex();
      int itower = eid.fiberChanId();
      int channelid = (itower & 0xF) | ((ilink & 0xF) << 4);
      int presamples = qiedf->presamples();

      if (!uhtrs.exist(uhtrIndex)) {
        uhtrs.newUHTR(uhtrIndex, presamples);
      }
      uhtrs.addChannel(uhtrIndex, qiedf, channelid, _verbosity);
    }
  }
  // -----------------------------------------------------
  // -----------------------------------------------------
  // loop over each uHTR and format data
  // -----------------------------------------------------
  // -----------------------------------------------------
  // loop over each uHTR and format data
  for (UHTRpacker::UHTRMap::iterator uhtr = uhtrs.uhtrs.begin(); uhtr != uhtrs.uhtrs.end(); ++uhtr) {
    uint64_t crateId = (uhtr->first) & 0xFF;
    uint64_t slotId = (uhtr->first & 0xF00) >> 8;

    uhtrs.finalizeHeadTail(&(uhtr->second), _verbosity);
    int fedId = FEDNumbering::MINHCALuTCAFEDID + crateId;
    if (fedMap.find(fedId) == fedMap.end()) {
      /* QUESTION: where should the orbit number come from? */
      fedMap[fedId] =
          std::make_unique<HCalFED>(fedId, iEvent.id().event(), iEvent.orbitNumber(), iEvent.bunchCrossing());
    }
    fedMap[fedId]->addUHTR(uhtr->second, crateId, slotId);
  }  // end loop over uhtr containers

  /* ------------------------------------------------------
     ------------------------------------------------------
           putting together the FEDRawDataCollection
     ------------------------------------------------------
     ------------------------------------------------------ */
  for (map<int, unique_ptr<HCalFED>>::iterator fed = fedMap.begin(); fed != fedMap.end(); ++fed) {
    int fedId = fed->first;

    auto& rawData = fed_buffers->FEDData(fedId);
    fed->second->formatFEDdata(rawData);

    FEDHeader hcalFEDHeader(rawData.data());
    hcalFEDHeader.set(rawData.data(), 1, iEvent.id().event(), iEvent.bunchCrossing(), fedId);
    FEDTrailer hcalFEDTrailer(rawData.data() + (rawData.size() - 8));
    hcalFEDTrailer.set(rawData.data() + (rawData.size() - 8),
                       rawData.size() / 8,
                       evf::compute_crc(rawData.data(), rawData.size()),
                       0,
                       0);

  }  // end loop over FEDs with data

  iEvent.put(std::move(fed_buffers));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalDigiToRawuHTR::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.addUntracked<int>("Verbosity", 0);
  desc.add<vector<int>>("tdc1", {8,  14, 15, 17, 8,  14, 15, 17, 8,  14, 14, 17, 8,  14, 14, 17, 8,  13, 14, 16, 8,  13,
                                 14, 16, 8,  12, 14, 15, 8,  12, 14, 15, 7,  12, 13, 15, 7,  12, 13, 15, 7,  12, 13, 15,
                                 7,  12, 13, 15, 7,  11, 12, 14, 7,  11, 12, 14, 7,  11, 12, 14, 7,  11, 12, 7});
  desc.add<vector<int>>("tdc2", {10, 16, 17, 19, 10, 16, 17, 19, 10, 16, 16, 19, 10, 16, 16, 19, 10, 15, 16, 18, 10, 15,
                                 16, 18, 10, 14, 16, 17, 10, 14, 16, 17, 9,  14, 15, 17, 9,  14, 15, 17, 9,  14, 15, 17,
                                 9,  14, 15, 17, 9,  13, 14, 16, 9,  13, 14, 16, 9,  13, 14, 16, 9,  13, 14, 9});
  desc.add<bool>("packHBTDC", true);
  desc.add<std::string>("ElectronicsMap", "");
  desc.add<edm::InputTag>("QIE10", edm::InputTag("simHcalDigis", "HFQIE10DigiCollection"));
  desc.add<edm::InputTag>("QIE11", edm::InputTag("simHcalDigis", "HBHEQIE11DigiCollection"));
  desc.add<edm::InputTag>("HBHEqie8", edm::InputTag("simHcalDigis"));
  desc.add<edm::InputTag>("HFqie8", edm::InputTag("simHcalDigis"));
  desc.add<edm::InputTag>("TP", edm::InputTag("simHcalTriggerPrimitiveDigis"));
  desc.add<bool>("premix", false);
  descriptions.add("hcalDigiToRawuHTR", desc);
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDigiToRawuHTR);
