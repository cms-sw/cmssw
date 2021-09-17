#include "ZdcSimpleReconstructor.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <iostream>

ZdcSimpleReconstructor::ZdcSimpleReconstructor(edm::ParameterSet const& conf)
    : reco_(conf.getParameter<bool>("correctForTimeslew"),
            conf.getParameter<bool>("correctForPhaseContainment"),
            conf.getParameter<double>("correctionPhaseNS"),
            conf.getParameter<int>("recoMethod"),
            conf.getParameter<int>("lowGainOffset"),
            conf.getParameter<double>("lowGainFrac")),
      det_(DetId::Hcal),
      dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")) {
  tok_input_castor = consumes<ZDCDigiCollection>(conf.getParameter<edm::InputTag>("digiLabelcastor"));
  tok_input_hcal = consumes<ZDCDigiCollection>(conf.getParameter<edm::InputTag>("digiLabelhcal"));

  std::string subd = conf.getParameter<std::string>("Subdetector");
  if (!strcasecmp(subd.c_str(), "ZDC")) {
    det_ = DetId::Calo;
    subdet_ = HcalZDCDetId::SubdetectorId;
    produces<ZDCRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(), "CALIB")) {
    subdet_ = HcalOther;
    subdetOther_ = HcalCalibration;
    produces<HcalCalibRecHitCollection>();
  } else {
    std::cout << "ZdcSimpleReconstructor is not associated with a specific subdetector!" << std::endl;
  }

  hcalTimeSlew_delay_ = nullptr;

  // ES tokens
  htopoToken_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  paramsToken_ = esConsumes<HcalLongRecoParams, HcalLongRecoParamsRcd, edm::Transition::BeginRun>();
  conditionsToken_ = esConsumes<HcalDbService, HcalDbRecord>();
  timeSlewToken_ = esConsumes<HcalTimeSlew, HcalTimeSlewRecord>(edm::ESInputTag("", "HBHE"));
}

ZdcSimpleReconstructor::~ZdcSimpleReconstructor() {}

void ZdcSimpleReconstructor::beginRun(edm::Run const& r, edm::EventSetup const& es) {
  const HcalTopology& htopo = es.getData(htopoToken_);
  const HcalLongRecoParams& p = es.getData(paramsToken_);
  longRecoParams_ = std::make_unique<HcalLongRecoParams>(p);
  longRecoParams_->setTopo(&htopo);

  hcalTimeSlew_delay_ = &es.getData(timeSlewToken_);
}

void ZdcSimpleReconstructor::endRun(edm::Run const& r, edm::EventSetup const& es) {}

void ZdcSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get conditions
  const HcalDbService* conditions = &eventSetup.getData(conditionsToken_);

  // define vectors to pass noiseTS and signalTS
  std::vector<unsigned int> mySignalTS;
  std::vector<unsigned int> myNoiseTS;

  if (det_ == DetId::Calo && subdet_ == HcalZDCDetId::SubdetectorId) {
    edm::Handle<ZDCDigiCollection> digi;
    e.getByToken(tok_input_hcal, digi);

    if (digi->empty()) {
      e.getByToken(tok_input_castor, digi);
      if (digi->empty())
        edm::LogInfo("ZdcHitReconstructor") << "No ZDC info found in either castorDigis or hcalDigis." << std::endl;
    }

    // create empty output
    auto rec = std::make_unique<ZDCRecHitCollection>();
    rec->reserve(digi->size());
    // run the algorithm
    unsigned int toaddMem = 0;

    ZDCDigiCollection::const_iterator i;
    for (i = digi->begin(); i != digi->end(); i++) {
      HcalZDCDetId cell = i->id();
      DetId detcell = (DetId)cell;
      // rof 27.03.09: drop ZS marked and passed digis:
      if (dropZSmarkedPassed_)
        if (i->zsMarkAndPass())
          continue;

      // get db values for signalTSs and noiseTSs
      const HcalLongRecoParam* myParams = longRecoParams_->getValues(detcell);
      mySignalTS.clear();
      myNoiseTS.clear();
      mySignalTS = myParams->signalTS();
      myNoiseTS = myParams->noiseTS();
      // warning: the PulseCorrection is not used by ZDC. If it gets a non-contingious set of
      // signal TS, it may not work properly. Assume contiguous here....
      unsigned int toadd = mySignalTS.size();
      if (toaddMem != toadd) {
        reco_.initPulseCorr(toadd, hcalTimeSlew_delay_);
        toaddMem = toadd;
      }
      const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
      const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
      HcalCoderDb coder(*channelCoder, *shape);
      rec->push_back(reco_.reconstruct(*i, myNoiseTS, mySignalTS, coder, calibrations));
    }
    // return result
    e.put(std::move(rec));
  }
}
