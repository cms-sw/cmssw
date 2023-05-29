/** \class EcalAnalFitUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes with the analytic specific fit method
 *   with alfa and beta fixed.
 *
 *  \author A. Ghezzi, Mar 2006
 *
 */

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitFixedAlphaBetaAlgo.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"

#include <cmath>
#include <fstream>
#include <iostream>

class EcalUncalibRecHitWorkerFixedAlphaBetaFit : public EcalUncalibRecHitWorkerRunOneDigiBase {
public:
  EcalUncalibRecHitWorkerFixedAlphaBetaFit(const edm::ParameterSet& ps, edm::ConsumesCollector&);
  EcalUncalibRecHitWorkerFixedAlphaBetaFit(){};
  ~EcalUncalibRecHitWorkerFixedAlphaBetaFit() override{};

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt,
           const EcalDigiCollection::const_iterator& digi,
           EcalUncalibratedRecHitCollection& result) override;

  edm::ParameterSetDescription getAlgoDescription() override;

private:
  double AmplThrEB_;
  double AmplThrEE_;

  EcalUncalibRecHitFixedAlphaBetaAlgo<EBDataFrame> algoEB_;
  EcalUncalibRecHitFixedAlphaBetaAlgo<EEDataFrame> algoEE_;

  double alphaEB_;
  double betaEB_;
  double alphaEE_;
  double betaEE_;
  std::vector<std::vector<std::pair<double, double> > >
      alphaBetaValues_;  // List of alpha and Beta values [SM#][CRY#](alpha, beta)
  bool useAlphaBetaArray_;
  std::string alphabetaFilename_;

  bool setAlphaBeta();  // Sets the alphaBetaValues_ vectors by the values provided in alphabetaFilename_

  edm::ESHandle<EcalGainRatios> pRatio;
  edm::ESHandle<EcalPedestals> pedHandle;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> ratiosToken_;
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalsToken_;
};

EcalUncalibRecHitWorkerFixedAlphaBetaFit::EcalUncalibRecHitWorkerFixedAlphaBetaFit(const edm::ParameterSet& ps,
                                                                                   edm::ConsumesCollector& c)
    : EcalUncalibRecHitWorkerRunOneDigiBase(ps, c),
      ratiosToken_(c.esConsumes<EcalGainRatios, EcalGainRatiosRcd>()),
      pedestalsToken_(c.esConsumes<EcalPedestals, EcalPedestalsRcd>()) {
  alphaEB_ = ps.getParameter<double>("alphaEB");
  betaEB_ = ps.getParameter<double>("betaEB");
  alphaEE_ = ps.getParameter<double>("alphaEE");
  betaEE_ = ps.getParameter<double>("betaEE");

  alphabetaFilename_ = ps.getUntrackedParameter<std::string>("AlphaBetaFilename");
  useAlphaBetaArray_ = setAlphaBeta();  // set crystalwise values of alpha and beta
  if (!useAlphaBetaArray_) {
    edm::LogInfo("EcalUncalibRecHitError") << " No alfa-beta file found. Using the deafult values.";
  }

  algoEB_.SetMinAmpl(ps.getParameter<double>("MinAmplBarrel"));
  algoEE_.SetMinAmpl(ps.getParameter<double>("MinAmplEndcap"));

  bool dyn_pede = ps.getParameter<bool>("UseDynamicPedestal");
  algoEB_.SetDynamicPedestal(dyn_pede);
  algoEE_.SetDynamicPedestal(dyn_pede);
}

void EcalUncalibRecHitWorkerFixedAlphaBetaFit::set(const edm::EventSetup& es) {
  // Gain Ratios
  LogDebug("EcalUncalibRecHitDebug") << "fetching gainRatios....";
  pRatio = es.getHandle(ratiosToken_);
  LogDebug("EcalUncalibRecHitDebug") << "done.";

  // fetch the pedestals from the cond DB via EventSetup
  LogDebug("EcalUncalibRecHitDebug") << "fetching pedestals....";
  pedHandle = es.getHandle(pedestalsToken_);
  LogDebug("EcalUncalibRecHitDebug") << "done.";
}

//Sets the alphaBetaValues_ vectors by the values provided in alphabetaFilename_
bool EcalUncalibRecHitWorkerFixedAlphaBetaFit::setAlphaBeta() {
  std::ifstream file(alphabetaFilename_.c_str());
  if (!file.is_open())
    return false;

  alphaBetaValues_.resize(36);

  char buffer[100];
  int sm, cry, ret;
  float a, b;
  std::pair<double, double> p(-1, -1);

  while (!file.getline(buffer, 100).eof()) {
    ret = sscanf(buffer, "%d %d %f %f", &sm, &cry, &a, &b);
    if ((ret != 4) || (sm <= 0) || (sm > 36) || (cry <= 0) || (cry > 1700)) {
      // send warning
      continue;
    }

    if (alphaBetaValues_[sm - 1].empty()) {
      alphaBetaValues_[sm - 1].resize(1700, p);
    }
    alphaBetaValues_[sm - 1][cry - 1].first = a;
    alphaBetaValues_[sm - 1][cry - 1].second = b;
  }

  file.close();
  return true;
}

bool EcalUncalibRecHitWorkerFixedAlphaBetaFit::run(const edm::Event& evt,
                                                   const EcalDigiCollection::const_iterator& itdg,
                                                   EcalUncalibratedRecHitCollection& result) {
  const EcalGainRatioMap& gainMap = pRatio.product()->getMap();  // map of gain ratios
  EcalGainRatioMap::const_iterator gainIter;                     // gain iterator
  EcalMGPAGainRatio aGain;                                       // gain object for a single xtal

  const EcalPedestalsMap& pedMap = pedHandle.product()->getMap();  // map of pedestals
  EcalPedestalsMapIterator pedIter;                                // pedestal iterator
  EcalPedestals::Item aped;                                        // pedestal object for a single xtal

  DetId detid(itdg->id());

  // find pedestals for this channel
  //LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << itdg->id();
  pedIter = pedMap.find(itdg->id());
  if (pedIter != pedMap.end()) {
    aped = (*pedIter);
  } else {
    edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << "error!! could not find pedestals for channel: ";
    if (detid.subdetId() == EcalBarrel) {
      edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << EBDetId(detid);
    } else {
      edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << EEDetId(detid);
    }
    edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << "\n  no uncalib rechit will be made for this digi!";
    return false;
  }
  double pedVec[3];
  pedVec[0] = aped.mean_x12;
  pedVec[1] = aped.mean_x6;
  pedVec[2] = aped.mean_x1;

  // find gain ratios
  //LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EBDetId(itdg->id()) ; // FIXME!!!!!!!!
  gainIter = gainMap.find(itdg->id());
  if (gainIter != gainMap.end()) {
    aGain = (*gainIter);
  } else {
    edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << "error!! could not find gain ratios for channel: ";
    if (detid.subdetId() == EcalBarrel) {
      edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << EBDetId(detid);
    } else {
      edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << EEDetId(detid);
    }
    edm::LogError("EcalUncalibRecHitWorkerFixedAlphaBetaFit") << "\n  no uncalib rechit will be made for this digi!";
    return false;
  }
  double gainRatios[3];
  gainRatios[0] = 1.;
  gainRatios[1] = aGain.gain12Over6();
  gainRatios[2] = aGain.gain6Over1() * aGain.gain12Over6();

  if (detid.subdetId() == EcalBarrel) {
    // Define Alpha and Beta either by stored values or by default universal values
    EBDetId ebDetId(detid);
    double a, b;
    if (useAlphaBetaArray_) {
      if (!alphaBetaValues_[ebDetId.ism() - 1].empty()) {
        a = alphaBetaValues_[ebDetId.ism() - 1][ebDetId.ic() - 1].first;
        b = alphaBetaValues_[ebDetId.ism() - 1][ebDetId.ic() - 1].second;
        if ((a == -1) && (b == -1)) {
          a = alphaEB_;
          b = betaEB_;
        }
      } else {
        a = alphaEB_;
        b = betaEB_;
      }
    } else {
      a = alphaEB_;
      b = betaEB_;
    }
    algoEB_.SetAlphaBeta(a, b);
    result.push_back(algoEB_.makeRecHit(*itdg, pedVec, gainRatios, nullptr, nullptr));
  } else {
    //FIX ME load in a and b from a file
    algoEE_.SetAlphaBeta(alphaEE_, betaEE_);
    result.push_back(algoEE_.makeRecHit(*itdg, pedVec, gainRatios, nullptr, nullptr));
  }
  return true;
}

edm::ParameterSetDescription EcalUncalibRecHitWorkerFixedAlphaBetaFit::getAlgoDescription() {
  edm::ParameterSetDescription psd;

  psd.addNode(edm::ParameterDescription<double>("alphaEB", 1.138, true) and
              edm::ParameterDescription<double>("alphaEE", 1.89, true) and
              edm::ParameterDescription<std::string>("AlphaBetaFilename", "NOFILE", false) and
              edm::ParameterDescription<double>("betaEB", 1.655, true) and
              edm::ParameterDescription<double>("MinAmplEndcap", 14.0, true) and
              edm::ParameterDescription<double>("MinAmplBarrel", 8.0, true) and
              edm::ParameterDescription<double>("betaEE", 1.4, true) and
              edm::ParameterDescription<bool>("UseDynamicPedestal", true, true));

  return psd;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitWorkerFactory,
                  EcalUncalibRecHitWorkerFixedAlphaBetaFit,
                  "EcalUncalibRecHitWorkerFixedAlphaBetaFit");
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitFillDescriptionWorkerFactory,
                  EcalUncalibRecHitWorkerFixedAlphaBetaFit,
                  "EcalUncalibRecHitWorkerFixedAlphaBetaFit");
