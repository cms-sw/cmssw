#include "HcalSimpleReconstructor.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <iostream>

HcalSimpleReconstructor::HcalSimpleReconstructor(edm::ParameterSet const& conf)
    : reco_(conf.getParameter<bool>("correctForTimeslew"),
            conf.getParameter<bool>("correctForPhaseContainment"),
            conf.getParameter<double>("correctionPhaseNS"),
            consumesCollector()),
      det_(DetId::Hcal),
      inputLabel_(conf.getParameter<edm::InputTag>("digiLabel")),
      dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
      firstSample_(conf.getParameter<int>("firstSample")),
      samplesToAdd_(conf.getParameter<int>("samplesToAdd")),
      tsFromDB_(conf.getParameter<bool>("tsFromDB")) {
  // register for data access
  tok_hf_ = consumes<HFDigiCollection>(inputLabel_);
  tok_ho_ = consumes<HODigiCollection>(inputLabel_);
  tok_calib_ = consumes<HcalCalibDigiCollection>(inputLabel_);

  std::string subd = conf.getParameter<std::string>("Subdetector");
  if (!strcasecmp(subd.c_str(), "HO")) {
    subdet_ = HcalOuter;
    produces<HORecHitCollection>();
  } else if (!strcasecmp(subd.c_str(), "HF")) {
    subdet_ = HcalForward;
    produces<HFRecHitCollection>();
  } else {
    std::cout << "HcalSimpleReconstructor is not associated with a specific subdetector!" << std::endl;
  }

  // ES tokens
  if (tsFromDB_) {
    htopoToken_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();
    paramsToken_ = esConsumes<HcalRecoParams, HcalRecoParamsRcd, edm::Transition::BeginRun>();
  }
  conditionsToken_ = esConsumes<HcalDbService, HcalDbRecord>();
}

HcalSimpleReconstructor::~HcalSimpleReconstructor() {}

void HcalSimpleReconstructor::beginRun(edm::Run const& r, edm::EventSetup const& es) {
  if (tsFromDB_) {
    const HcalTopology& htopo = es.getData(htopoToken_);
    const HcalRecoParams& p = es.getData(paramsToken_);
    paramTS_ = std::make_unique<HcalRecoParams>(p);
    paramTS_->setTopo(&htopo);
  }
  reco_.beginRun(es);
}

void HcalSimpleReconstructor::endRun(edm::Run const& r, edm::EventSetup const& es) { reco_.endRun(); }

template <class DIGICOLL, class RECHITCOLL>
void HcalSimpleReconstructor::process(edm::Event& e,
                                      const edm::EventSetup& eventSetup,
                                      const edm::EDGetTokenT<DIGICOLL>& tok) {
  // get conditions
  const HcalDbService* conditions = &eventSetup.getData(conditionsToken_);

  edm::Handle<DIGICOLL> digi;
  e.getByToken(tok, digi);

  // create empty output
  auto rec = std::make_unique<RECHITCOLL>();
  rec->reserve(digi->size());
  // run the algorithm
  int first = firstSample_;
  int toadd = samplesToAdd_;
  typename DIGICOLL::const_iterator i;
  for (i = digi->begin(); i != digi->end(); i++) {
    HcalDetId cell = i->id();
    DetId detcell = (DetId)cell;
    // rof 27.03.09: drop ZS marked and passed digis:
    if (dropZSmarkedPassed_)
      if (i->zsMarkAndPass())
        continue;

    const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
    const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
    const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
    HcalCoderDb coder(*channelCoder, *shape);

    //>>> firstSample & samplesToAdd
    if (tsFromDB_) {
      const HcalRecoParam* param_ts = paramTS_->getValues(detcell.rawId());
      first = param_ts->firstSample();
      toadd = param_ts->samplesToAdd();
    }
    rec->push_back(reco_.reconstruct(*i, first, toadd, coder, calibrations));
  }
  // return result
  e.put(std::move(rec));
}

void HcalSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  if (det_ == DetId::Hcal) {
    if (subdet_ == HcalForward) {
      process<HFDigiCollection, HFRecHitCollection>(e, eventSetup, tok_hf_);
    } else if (subdet_ == HcalOuter) {
      process<HODigiCollection, HORecHitCollection>(e, eventSetup, tok_ho_);
    } else if (subdet_ == HcalOther && subdetOther_ == HcalCalibration) {
      process<HcalCalibDigiCollection, HcalCalibRecHitCollection>(e, eventSetup, tok_calib_);
    }
  }
}

void HcalSimpleReconstructor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // horeco
  edm::ParameterSetDescription descHO;
  descHO.add<double>("correctionPhaseNS", 13.0);
  descHO.add<edm::InputTag>("digiLabel", edm::InputTag("hcalDigis"));
  descHO.add<bool>("tsFromDB", true);
  descHO.add<int>("samplesToAdd", 4);
  descHO.add<std::string>("Subdetector", "HO");
  descHO.add<bool>("correctForTimeslew", true);
  descHO.add<bool>("dropZSmarkedPassed", true);
  descHO.add<bool>("correctForPhaseContainment", true);
  descHO.add<int>("firstSample", 4);
  descriptions.add("hosimplereco", descHO);

  // hfreco
  edm::ParameterSetDescription descHF;
  descHF.add<double>("correctionPhaseNS", 0.0);
  descHF.add<edm::InputTag>("digiLabel", edm::InputTag("hcalDigis"));
  descHF.add<bool>("tsFromDB", true);
  descHF.add<int>("samplesToAdd", 2);
  descHF.add<std::string>("Subdetector", "HF");
  descHF.add<bool>("correctForTimeslew", false);
  descHF.add<bool>("dropZSmarkedPassed", true);
  descHF.add<bool>("correctForPhaseContainment", false);
  descHF.add<int>("firstSample", 4);
  descriptions.add("hfsimplereco", descHF);
}
