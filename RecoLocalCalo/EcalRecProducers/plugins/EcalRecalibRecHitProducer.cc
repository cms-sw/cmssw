/** \class EcalRecalibRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  \author Federico Ferri, University of Milano Bicocca and INFN
 *
 **/

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"

#include <cmath>
#include <iostream>
#include <vector>

class EcalRecalibRecHitProducer : public edm::global::EDProducer<> {
public:
  explicit EcalRecalibRecHitProducer(const edm::ParameterSet& ps);
  void produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& es) const override;

private:
  const edm::InputTag EBRecHitCollection_;
  const edm::InputTag EERecHitCollection_;
  const edm::EDGetTokenT<EBRecHitCollection> EBRecHitToken_;
  const edm::EDGetTokenT<EERecHitCollection> EERecHitToken_;

  const std::string EBRecalibRecHitCollection_;  // secondary name to be given to EB collection of hits
  const std::string EERecalibRecHitCollection_;  // secondary name to be given to EE collection of hits

  const bool doEnergyScale_;
  const bool doIntercalib_;
  const bool doLaserCorrections_;
  const bool doEnergyScaleInverse_;
  const bool doIntercalibInverse_;
  const bool doLaserCorrectionsInverse_;

  edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> ecalADCToGeVConstantToken_;
  edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> ecalIntercalibConstantsToken_;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> ecalLaserDBServiceToken_;
};

EcalRecalibRecHitProducer::EcalRecalibRecHitProducer(const edm::ParameterSet& ps)
    : EBRecHitCollection_(ps.getParameter<edm::InputTag>("EBRecHitCollection")),
      EERecHitCollection_(ps.getParameter<edm::InputTag>("EERecHitCollection")),
      EBRecHitToken_((not EBRecHitCollection_.label().empty()) ? consumes<EBRecHitCollection>(EBRecHitCollection_)
                                                               : edm::EDGetTokenT<EBRecHitCollection>()),
      EERecHitToken_((not EERecHitCollection_.label().empty()) ? consumes<EERecHitCollection>(EERecHitCollection_)
                                                               : edm::EDGetTokenT<EERecHitCollection>()),
      EBRecalibRecHitCollection_(ps.getParameter<std::string>("EBRecalibRecHitCollection")),
      EERecalibRecHitCollection_(ps.getParameter<std::string>("EERecalibRecHitCollection")),
      doEnergyScale_(ps.getParameter<bool>("doEnergyScale")),
      doIntercalib_(ps.getParameter<bool>("doIntercalib")),
      doLaserCorrections_(ps.getParameter<bool>("doLaserCorrections")),

      doEnergyScaleInverse_(ps.getParameter<bool>("doEnergyScaleInverse")),
      doIntercalibInverse_(ps.getParameter<bool>("doIntercalibInverse")),
      doLaserCorrectionsInverse_(ps.getParameter<bool>("doLaserCorrectionsInverse")),
      ecalLaserDBServiceToken_(esConsumes<EcalLaserDbService, EcalLaserDbRecord>()) {
  if (doEnergyScale_) {
    ecalADCToGeVConstantToken_ = esConsumes<EcalADCToGeVConstant, EcalADCToGeVConstantRcd>();
  }
  if (doIntercalib_) {
    ecalIntercalibConstantsToken_ = esConsumes<EcalIntercalibConstants, EcalIntercalibConstantsRcd>();
  }
  produces<EBRecHitCollection>(EBRecalibRecHitCollection_);
  produces<EERecHitCollection>(EERecalibRecHitCollection_);
}

void EcalRecalibRecHitProducer::produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& es) const {
  using namespace edm;
  Handle<EBRecHitCollection> pEBRecHits;
  Handle<EERecHitCollection> pEERecHits;

  const EBRecHitCollection* EBRecHits = nullptr;
  const EERecHitCollection* EERecHits = nullptr;

  if (not EBRecHitCollection_.label().empty()) {
    evt.getByToken(EBRecHitToken_, pEBRecHits);
    EBRecHits = pEBRecHits.product();  // get a ptr to the product
  }
  if (not EERecHitCollection_.label().empty()) {
    evt.getByToken(EERecHitToken_, pEERecHits);
    EERecHits = pEERecHits.product();  // get a ptr to the product
  }

  // collection of rechits to put in the event
  auto EBRecalibRecHits = std::make_unique<EBRecHitCollection>();
  auto EERecalibRecHits = std::make_unique<EERecHitCollection>();

  // now fetch all conditions we need to make rechits
  // ADC to GeV constant
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  const EcalADCToGeVConstant* agc = nullptr;
  float agc_eb = 1.;
  float agc_ee = 1.;
  if (doEnergyScale_) {
    pAgc = es.getHandle(ecalADCToGeVConstantToken_);
    agc = pAgc.product();
    // use this value in the algorithm
    agc_eb = float(agc->getEBValue());
    agc_ee = float(agc->getEEValue());
  }
  // Intercalib constants
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  const EcalIntercalibConstants* ical = nullptr;
  if (doIntercalib_) {
    pIcal = es.getHandle(ecalIntercalibConstantsToken_);
    ical = pIcal.product();
  }
  // Laser corrections
  edm::ESHandle<EcalLaserDbService> pLaser = es.getHandle(ecalLaserDBServiceToken_);

  if (doEnergyScaleInverse_) {
    agc_eb = 1.0 / agc_eb;
    agc_ee = 1.0 / agc_ee;
  }

  if (EBRecHits) {
    // loop over uncalibrated rechits to make calibrated ones
    for (EBRecHitCollection::const_iterator it = EBRecHits->begin(); it != EBRecHits->end(); ++it) {
      EcalIntercalibConstant icalconst = 1.;
      if (doIntercalib_) {
        // find intercalib constant for this xtal
        const EcalIntercalibConstantMap& icalMap = ical->getMap();
        EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(it->id());
        if (icalit != icalMap.end()) {
          icalconst = (*icalit);
        } else {
          edm::LogError("EcalRecHitError") << "No intercalib const found for xtal " << EBDetId(it->id())
                                           << "! something wrong with EcalIntercalibConstants in your DB? ";
        }
      }
      // get laser coefficient
      float lasercalib = 1;
      if (doLaserCorrections_) {
        lasercalib = pLaser->getLaserCorrection(EBDetId(it->id()), evt.time());
      }

      // make the rechit and put in the output collection
      // must implement op= for EcalRecHit

      if (doIntercalibInverse_) {
        icalconst = 1.0 / icalconst;
      }
      if (doLaserCorrectionsInverse_) {
        lasercalib = 1.0 / lasercalib;
      }

      EcalRecHit aHit((*it).id(), (*it).energy() * agc_eb * icalconst * lasercalib, (*it).time());
      EBRecalibRecHits->push_back(aHit);
    }
  }

  if (EERecHits) {
    // loop over uncalibrated rechits to make calibrated ones
    for (EERecHitCollection::const_iterator it = EERecHits->begin(); it != EERecHits->end(); ++it) {
      // find intercalib constant for this xtal
      EcalIntercalibConstant icalconst = 1.;
      if (doIntercalib_) {
        const EcalIntercalibConstantMap& icalMap = ical->getMap();
        EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(it->id());
        if (icalit != icalMap.end()) {
          icalconst = (*icalit);
        } else {
          edm::LogError("EcalRecHitError") << "No intercalib const found for xtal " << EEDetId(it->id())
                                           << "! something wrong with EcalIntercalibConstants in your DB? ";
        }
      }
      // get laser coefficient
      float lasercalib = 1;
      if (doLaserCorrections_) {
        lasercalib = pLaser->getLaserCorrection(EEDetId(it->id()), evt.time());
      }

      if (doIntercalibInverse_) {
        icalconst = 1.0 / icalconst;
      }
      if (doLaserCorrectionsInverse_) {
        lasercalib = 1.0 / lasercalib;
      }

      // make the rechit and put in the output collection
      EcalRecHit aHit((*it).id(), (*it).energy() * agc_ee * icalconst * lasercalib, (*it).time());
      EERecalibRecHits->push_back(aHit);
    }
  }
  // put the collection of recunstructed hits in the event
  LogInfo("EcalRecalibRecHitInfo") << "total # EB re-calibrated rechits: " << EBRecalibRecHits->size();
  LogInfo("EcalRecalibRecHitInfo") << "total # EE re-calibrated rechits: " << EERecalibRecHits->size();

  evt.put(std::move(EBRecalibRecHits), EBRecalibRecHitCollection_);
  evt.put(std::move(EERecalibRecHits), EERecalibRecHitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalRecalibRecHitProducer);
