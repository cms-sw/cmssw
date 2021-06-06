#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

#include "EventFilter/HcalRawToDigi/plugins/HcalDigiToRaw.h"

using namespace std;

HcalDigiToRaw::HcalDigiToRaw(edm::ParameterSet const& conf)
    : hbheTag_(conf.getUntrackedParameter("HBHE", edm::InputTag())),
      hoTag_(conf.getUntrackedParameter("HO", edm::InputTag())),
      hfTag_(conf.getUntrackedParameter("HF", edm::InputTag())),
      zdcTag_(conf.getUntrackedParameter("ZDC", edm::InputTag())),
      calibTag_(conf.getUntrackedParameter("CALIB", edm::InputTag())),
      trigTag_(conf.getUntrackedParameter("TRIG", edm::InputTag())),
      // register for data access
      tok_hbhe_(consumes<HBHEDigiCollection>(hbheTag_)),
      tok_ho_(consumes<HODigiCollection>(hoTag_)),
      tok_hf_(consumes<HFDigiCollection>(hfTag_)),
      tok_calib_(consumes<HcalCalibDigiCollection>(calibTag_)),
      tok_zdc_(consumes<ZDCDigiCollection>(zdcTag_)),
      tok_htp_(consumes<HcalTrigPrimDigiCollection>(trigTag_)),
      tok_dbService_(esConsumes<HcalDbService, HcalDbRecord>()) {
  produces<FEDRawDataCollection>();
}

// Virtual destructor needed.
HcalDigiToRaw::~HcalDigiToRaw() {}

// Functions that gets called by framework every event
void HcalDigiToRaw::produce(edm::StreamID id, edm::Event& e, const edm::EventSetup& es) const {
  HcalPacker::Collections colls;

  // Step A: Get Inputs
  edm::Handle<HBHEDigiCollection> hbhe;
  if (!hbheTag_.label().empty()) {
    e.getByToken(tok_hbhe_, hbhe);
    colls.hbhe = hbhe.product();
  }
  edm::Handle<HODigiCollection> ho;
  if (!hoTag_.label().empty()) {
    e.getByToken(tok_ho_, ho);
    colls.hoCont = ho.product();
  }
  edm::Handle<HFDigiCollection> hf;
  if (!hfTag_.label().empty()) {
    e.getByToken(tok_hf_, hf);
    colls.hfCont = hf.product();
  }
  edm::Handle<HcalCalibDigiCollection> Calib;
  if (!calibTag_.label().empty()) {
    e.getByToken(tok_calib_, Calib);
    colls.calibCont = Calib.product();
  }
  edm::Handle<ZDCDigiCollection> zdc;
  if (!zdcTag_.label().empty()) {
    e.getByToken(tok_zdc_, zdc);
    colls.zdcCont = zdc.product();
  }
  edm::Handle<HcalTrigPrimDigiCollection> htp;
  if (!trigTag_.label().empty()) {
    e.getByToken(tok_htp_, htp);
    if (htp.isValid())
      colls.tpCont = htp.product();
  }
  // get the mapping
  edm::ESHandle<HcalDbService> pSetup = es.getHandle(tok_dbService_);
  const HcalElectronicsMap* readoutMap = pSetup->getHcalMapping();
  // Step B: Create empty output
  auto raw = std::make_unique<FEDRawDataCollection>();

  const int ifed_first = FEDNumbering::MINHCALFEDID;
  const int ifed_last = FEDNumbering::MAXHCALFEDID;

  int orbitN = e.id().event();
  int bcnN = 2000;

  // Step C: pack all requested FEDs
  for (int ifed = ifed_first; ifed <= ifed_last; ++ifed) {
    FEDRawData& fed = raw->FEDData(ifed);
    try {
      packer_.pack(ifed, ifed - ifed_first, e.id().event(), orbitN, bcnN, colls, *readoutMap, fed);
    } catch (cms::Exception& e) {
      edm::LogWarning("Unpacking error") << e.what();
    } catch (...) {
      edm::LogWarning("Unpacking exception");
    }
  }

  e.put(std::move(raw));
}
