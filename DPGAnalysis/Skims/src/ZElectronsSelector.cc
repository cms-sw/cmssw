// -*- C++ -*-
// Class:      ZElectronsSelector
//
// Original Author:  Silvia Taroni
//         Created:  Wed, 29 Nov 2017 18:23:54 GMT
//
//

#include "FWCore/PluginManager/interface/ModuleDef.h"

// system include files

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
//
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/Common/interface/View.h"

using namespace std;
using namespace reco;
namespace edm {
  class EventSetup;
}

class ZElectronsSelector {
public:
  ZElectronsSelector(const edm::ParameterSet&, edm::ConsumesCollector& iC);
  bool operator()(const reco::GsfElectron&) const;
  void newEvent(const edm::Event&, const edm::EventSetup&);
  const float getEffectiveArea(float eta) const;
  void printEffectiveAreas() const;

  edm::EDGetTokenT<double> theRhoToken;
  edm::EDGetTokenT<reco::GsfElectronCollection> theGsfEToken;
  edm::Handle<double> _rhoHandle;

  std::vector<double> absEtaMin_;            // low limit of the eta range
  std::vector<double> absEtaMax_;            // upper limit of the eta range
  std::vector<double> effectiveAreaValues_;  // effective area for this eta range

  edm::ParameterSet eleIDWP;

  vector<int> missHits;
  vector<double> sigmaIEtaIEtaCut;
  vector<double> dEtaInSeedCut;
  vector<double> dPhiInCut;
  vector<double> hOverECut;
  vector<double> relCombIso;
  vector<double> EInvMinusPInv;
};

void ZElectronsSelector::printEffectiveAreas() const {
  printf("  eta_min   eta_max    effective area\n");
  uint nEtaBins = absEtaMin_.size();
  for (uint iEta = 0; iEta < nEtaBins; iEta++) {
    printf("  %8.4f    %8.4f   %8.5f\n", absEtaMin_[iEta], absEtaMax_[iEta], effectiveAreaValues_[iEta]);
  }
}
const float ZElectronsSelector::getEffectiveArea(float eta) const {
  float effArea = 0;
  uint nEtaBins = absEtaMin_.size();
  for (uint iEta = 0; iEta < nEtaBins; iEta++) {
    if (std::abs(eta) >= absEtaMin_[iEta] && std::abs(eta) < absEtaMax_[iEta]) {
      effArea = effectiveAreaValues_[iEta];
      break;
    }
  }

  return effArea;
}

ZElectronsSelector::ZElectronsSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : theRhoToken(iC.consumes<double>(cfg.getParameter<edm::InputTag>("rho"))) {
  absEtaMin_ = cfg.getParameter<std::vector<double> >("absEtaMin");
  absEtaMax_ = cfg.getParameter<std::vector<double> >("absEtaMax");
  effectiveAreaValues_ = cfg.getParameter<std::vector<double> >("effectiveAreaValues");
  //printEffectiveAreas();
  eleIDWP = cfg.getParameter<edm::ParameterSet>("eleID");

  missHits = eleIDWP.getParameter<std::vector<int> >("missingHitsCut");
  sigmaIEtaIEtaCut = eleIDWP.getParameter<std::vector<double> >("full5x5_sigmaIEtaIEtaCut");
  dEtaInSeedCut = eleIDWP.getParameter<std::vector<double> >("dEtaInSeedCut");
  dPhiInCut = eleIDWP.getParameter<std::vector<double> >("dPhiInCut");
  hOverECut = eleIDWP.getParameter<std::vector<double> >("hOverECut");
  relCombIso = eleIDWP.getParameter<std::vector<double> >("relCombIsolationWithEACut");
  EInvMinusPInv = eleIDWP.getParameter<std::vector<double> >("EInverseMinusPInverseCut");
}

void ZElectronsSelector::newEvent(const edm::Event& ev, const edm::EventSetup&) {
  ev.getByToken(theRhoToken, _rhoHandle);
}

bool ZElectronsSelector::operator()(const reco::GsfElectron& el) const {
  float pt_e = el.pt();
  unsigned int ind = 0;
  auto etrack = el.gsfTrack();
  float abseta = fabs((el.superCluster().get())->position().eta());

  if (el.isEB()) {
    if (abseta > 1.479)
      return false;  // check if it is really needed
  }
  if (el.isEE()) {
    ind = 1;
    if (abseta < 1.479)
      return false;  // check if it is really needed
    if (abseta >= 2.5)
      return false;  // check if it is really needed
  }

  if (etrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) > missHits[ind])
    return false;
  if (el.full5x5_sigmaIetaIeta() > sigmaIEtaIEtaCut[ind])
    return false;
  if (fabs(el.deltaPhiSuperClusterTrackAtVtx()) > dPhiInCut[ind])
    return false;
  if (fabs(el.deltaEtaSeedClusterTrackAtVtx()) > dEtaInSeedCut[ind])
    return false;
  if (el.hadronicOverEm() > hOverECut[ind])
    return false;
  const float eA = getEffectiveArea(abseta);
  const float rho = _rhoHandle.isValid() ? (float)(*_rhoHandle.product()) : 0;
  if ((el.pfIsolationVariables().sumChargedHadronPt +
       std::max(float(0.0),
                el.pfIsolationVariables().sumNeutralHadronEt + el.pfIsolationVariables().sumPhotonEt - eA * rho)) >
      relCombIso[ind] * pt_e)
    return false;
  const float ecal_energy_inverse = 1.0 / el.ecalEnergy();
  const float eSCoverP = el.eSuperClusterOverP();
  if (std::abs(1.0 - eSCoverP) * ecal_energy_inverse > EInvMinusPInv[ind])
    return false;

  return true;
}

EVENTSETUP_STD_INIT(ZElectronsSelector);

typedef SingleObjectSelector<edm::View<reco::GsfElectron>, ZElectronsSelector> ZElectronsSelectorAndSkim;

DEFINE_FWK_MODULE(ZElectronsSelectorAndSkim);
