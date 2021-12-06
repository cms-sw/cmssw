#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorChannelStatus.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParams.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorrs.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoLocalCalo/CastorReco/interface/CastorSimpleRecAlgo.h"

#include <iostream>

class CastorSimpleReconstructor : public edm::stream::EDProducer<> {
public:
  explicit CastorSimpleReconstructor(const edm::ParameterSet& ps);
  ~CastorSimpleReconstructor() override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  CastorSimpleRecAlgo reco_;
  DetId::Detector det_;
  int subdet_;
  //      HcalOtherSubdetector subdetOther_;
  edm::EDGetTokenT<CastorDigiCollection> tok_input_;
  edm::ESGetToken<CastorDbService, CastorDbRecord> tok_conditions_;
  edm::ESGetToken<CastorRecoParams, CastorRecoParamsRcd> tok_recoParams_;
  edm::ESGetToken<CastorSaturationCorrs, CastorSaturationCorrsRcd> tok_satCorr_;

  int firstSample_;
  int samplesToAdd_;
  int maxADCvalue_;
  bool tsFromDB_;
  bool setSaturationFlag_;
  bool doSaturationCorr_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CastorSimpleReconstructor);

using namespace std;

CastorSimpleReconstructor::CastorSimpleReconstructor(edm::ParameterSet const& conf)
    : reco_(conf.getParameter<int>("firstSample"),
            conf.getParameter<int>("samplesToAdd"),
            conf.getParameter<bool>("correctForTimeslew"),
            conf.getParameter<bool>("correctForPhaseContainment"),
            conf.getParameter<double>("correctionPhaseNS")),
      det_(DetId::Hcal),
      firstSample_(conf.getParameter<int>("firstSample")),
      samplesToAdd_(conf.getParameter<int>("samplesToAdd")),
      maxADCvalue_(conf.getParameter<int>("maxADCvalue")),
      tsFromDB_(conf.getParameter<bool>("tsFromDB")),
      setSaturationFlag_(conf.getParameter<bool>("setSaturationFlag")),
      doSaturationCorr_(conf.getParameter<bool>("doSaturationCorr")) {
  tok_input_ = consumes<CastorDigiCollection>(conf.getParameter<edm::InputTag>("digiLabel"));
  tok_conditions_ = esConsumes<CastorDbService, CastorDbRecord>();

  std::string subd = conf.getParameter<std::string>("Subdetector");
  if (!strcasecmp(subd.c_str(), "CASTOR")) {
    det_ = DetId::Calo;
    subdet_ = HcalCastorDetId::SubdetectorId;
    produces<CastorRecHitCollection>();
  } else {
    edm::LogWarning("CastorSimpleReconstructor")
        << "CastorSimpleReconstructor is not associated with CASTOR subdetector!" << std::endl;
  }
  if (tsFromDB_) {
    tok_recoParams_ = esConsumes<CastorRecoParams, CastorRecoParamsRcd>();
  }
  if (doSaturationCorr_) {
    tok_satCorr_ = esConsumes<CastorSaturationCorrs, CastorSaturationCorrsRcd>();
  }
}

CastorSimpleReconstructor::~CastorSimpleReconstructor() {}

void CastorSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get conditions
  edm::ESHandle<CastorDbService> conditions = eventSetup.getHandle(tok_conditions_);
  const CastorQIEShape* shape = conditions->getCastorShape();  // this one is generic

  CastorCalibrations calibrations;

  // try to get the TS windows from the db
  edm::ESHandle<CastorRecoParams> recoparams;
  if (tsFromDB_) {
    recoparams = eventSetup.getHandle(tok_recoParams_);
    if (!recoparams.isValid()) {
      tsFromDB_ = false;
      edm::LogWarning("CastorSimpleReconstructor")
          << "Could not handle the CastorRecoParamsRcd correctly, using parameters from cfg file from this event "
             "onwards... These parameters could be wrong for this run... please check"
          << std::endl;
    }
  }

  // try to get the saturation correction constants from the db
  edm::ESHandle<CastorSaturationCorrs> satcorr;
  if (doSaturationCorr_) {
    satcorr = eventSetup.getHandle(tok_satCorr_);
    if (!satcorr.isValid()) {
      doSaturationCorr_ = false;
      edm::LogWarning("CastorSimpleReconstructor") << "Could not handle the CastorSaturationCorrsRcd correctly. We'll "
                                                      "not try the saturation correction from this event onwards..."
                                                   << std::endl;
    }
  }

  if (det_ == DetId::Calo && subdet_ == HcalCastorDetId::SubdetectorId) {
    edm::Handle<CastorDigiCollection> digi;
    e.getByToken(tok_input_, digi);

    // create empty output
    auto rec = std::make_unique<CastorRecHitCollection>();
    // run the algorithm
    CastorDigiCollection::const_iterator i;
    for (i = digi->begin(); i != digi->end(); i++) {
      HcalCastorDetId cell = i->id();
      DetId detcell = (DetId)cell;
      const CastorCalibrations& calibrations = conditions->getCastorCalibrations(cell);

      if (tsFromDB_) {
        const CastorRecoParam* param_ts = recoparams->getValues(detcell.rawId());
        reco_.resetTimeSamples(param_ts->firstSample(), param_ts->samplesToAdd());
      }
      const CastorQIECoder* channelCoder = conditions->getCastorCoder(cell);
      CastorCoderDb coder(*channelCoder, *shape);

      // reconstruct the rechit
      rec->push_back(reco_.reconstruct(*i, coder, calibrations));

      // set the saturation flag if needed
      if (setSaturationFlag_) {
        reco_.checkADCSaturation(rec->back(), *i, maxADCvalue_);

        //++++ Saturation Correction +++++
        if (doSaturationCorr_ && rec->back().flagField(HcalCaloFlagLabels::ADCSaturationBit)) {
          // get saturation correction value
          const CastorSaturationCorr* saturationCorr = satcorr->getValues(detcell.rawId());
          double satCorrConst = 1.;
          satCorrConst = saturationCorr->getValue();
          reco_.recoverADCSaturation(rec->back(), coder, calibrations, *i, maxADCvalue_, satCorrConst);
        }
      }
    }
    // return result
    e.put(std::move(rec));
  }
}
