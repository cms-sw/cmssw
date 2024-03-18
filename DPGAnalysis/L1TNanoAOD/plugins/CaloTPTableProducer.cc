// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      CaloTPTableProducer
//
/**\class CaloTPTableProducer CaloTPTableProducer.cc PhysicsTools/NanoAOD/plugins/CaloTPTableProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  localusers user
//         Created:  Wed, 08 Nov 2023 13:16:40 GMT
//
//

// system include files
#include <memory>

// user include files
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//
// class declaration
//

class CaloTPTableProducer : public edm::stream::EDProducer<> {
public:
  explicit CaloTPTableProducer(const edm::ParameterSet&);
  ~CaloTPTableProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  const double ecalLSB_;

  const edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPsToken_;
  const std::string ecalTPsName_;

  const edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPsToken_;
  const std::string hcalTPsName_;

  const edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoderToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloTPTableProducer::CaloTPTableProducer(const edm::ParameterSet& iConfig)
    : ecalLSB_(iConfig.getUntrackedParameter<double>("ecalLSB", 0.5)),
      ecalTPsToken_(consumes<EcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalTPsSrc"))),
      ecalTPsName_(iConfig.getParameter<std::string>("ecalTPsName")),
      hcalTPsToken_(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalTPsSrc"))),
      hcalTPsName_(iConfig.getParameter<std::string>("hcalTPsName")),
      decoderToken_(esConsumes<CaloTPGTranscoder, CaloTPGRecord>()) {
  produces<nanoaod::FlatTable>("EcalTP");
  produces<nanoaod::FlatTable>("HcalTP");

  //now do what ever other initialization is needed
}

CaloTPTableProducer::~CaloTPTableProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CaloTPTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<CaloTPGTranscoder> decoder;
  decoder = iSetup.getHandle(decoderToken_);

  edm::Handle<EcalTrigPrimDigiCollection> ecalTPs;
  iEvent.getByToken(ecalTPsToken_, ecalTPs);

  edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
  iEvent.getByToken(hcalTPsToken_, hcalTPs);

  vector<int> ecalTPieta;
  vector<int> ecalTPCaliphi;
  vector<int> ecalTPiphi;
  vector<float> ecalTPet;
  vector<int> ecalTPcompEt;
  vector<int> ecalTPfineGrain;
  int nECALTP(0);
  if (ecalTPs.isValid()) {
    for (const auto& itr : *(ecalTPs.product())) {
      short ieta = (short)itr.id().ieta();

      unsigned short cal_iphi = (unsigned short)itr.id().iphi();
      unsigned short iphi = (72 + 18 - cal_iphi) % 72;
      unsigned short compEt = itr.compressedEt();
      double et = ecalLSB_ * compEt;
      unsigned short fineGrain = (unsigned short)itr.fineGrain();

      if (compEt > 0) {
        ecalTPieta.push_back(ieta);
        ecalTPCaliphi.push_back(cal_iphi);
        ecalTPiphi.push_back(iphi);
        ecalTPet.push_back(et);
        ecalTPcompEt.push_back(compEt);
        ecalTPfineGrain.push_back(fineGrain);
        nECALTP++;
      }
    }
  }
  auto ecalTPTable = std::make_unique<nanoaod::FlatTable>(nECALTP, ecalTPsName_, false);
  ecalTPTable->addColumn<int16_t>("ieta", ecalTPieta, "");
  ecalTPTable->addColumn<int16_t>("Caliphi", ecalTPCaliphi, "");
  ecalTPTable->addColumn<int16_t>("iphi", ecalTPiphi, "");
  ecalTPTable->addColumn<float>("et", ecalTPet, "", 12);
  ecalTPTable->addColumn<int16_t>("compEt", ecalTPcompEt, "");
  ecalTPTable->addColumn<int16_t>("fineGrain", ecalTPfineGrain, "");

  vector<int> hcalTPieta;
  vector<int> hcalTPCaliphi;
  vector<int> hcalTPiphi;
  vector<float> hcalTPet;
  vector<int> hcalTPcompEt;
  vector<int> hcalTPfineGrain;
  int nHCALTP(0);
  if (hcalTPs.isValid()) {
    for (auto itr : (*hcalTPs.product())) {
      int ver = itr.id().version();
      short ieta = (short)itr.id().ieta();
      unsigned short absIeta = (unsigned short)abs(ieta);
      unsigned short cal_iphi = (unsigned short)itr.id().iphi();
      unsigned short iphi = (72 + 18 - cal_iphi) % 72;

      unsigned short compEt = itr.SOI_compressedEt();
      double et = decoder->hcaletValue(itr.id(), itr.t0());
      unsigned short fineGrain = (unsigned short)itr.SOI_fineGrain();

      if (compEt > 0 && (absIeta < 29 || ver == 1)) {
        hcalTPieta.push_back(ieta);
        hcalTPCaliphi.push_back(cal_iphi);
        hcalTPiphi.push_back(iphi);
        hcalTPet.push_back(et);
        hcalTPcompEt.push_back(compEt);
        hcalTPfineGrain.push_back(fineGrain);
        nHCALTP++;
      }
    }
  }

  auto hcalTPTable = std::make_unique<nanoaod::FlatTable>(nHCALTP, hcalTPsName_, false);
  hcalTPTable->addColumn<int16_t>("ieta", hcalTPieta, "");
  hcalTPTable->addColumn<int16_t>("Caliphi", hcalTPCaliphi, "");
  hcalTPTable->addColumn<int16_t>("iphi", hcalTPiphi, "");
  hcalTPTable->addColumn<float>("et", hcalTPet, "", 12);
  hcalTPTable->addColumn<int16_t>("compEt", hcalTPcompEt, "");
  hcalTPTable->addColumn<int16_t>("fineGrain", hcalTPfineGrain, "");

  iEvent.put(std::move(ecalTPTable), "HcalTP");
  iEvent.put(std::move(hcalTPTable), "EcalTP");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void CaloTPTableProducer::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void CaloTPTableProducer::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
CaloTPTableProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
CaloTPTableProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CaloTPTableProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CaloTPTableProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CaloTPTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  
  desc.add<std::string>("name", "l1calotowerflattableproducer");
  desc.addUntracked<double>("ecalLSB", 0.5);
  desc.add<edm::InputTag>("ecalTPsSrc", edm::InputTag{"ecalDigis","EcalTriggerPrimitives"});
  desc.add<string>("ecalTPsName", "EcalUnpackedTPs");
  desc.add<edm::InputTag>("hcalTPsSrc", edm::InputTag{"hcalDigis"});
  desc.add<string>("hcalTPsName", "HcalUnpackedTPs");

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloTPTableProducer);
