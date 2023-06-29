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

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
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

class IsoPhotonEBSelector {
public:
  IsoPhotonEBSelector(const edm::ParameterSet&, edm::ConsumesCollector& iC);
  bool operator()(const reco::Photon&) const;
  void newEvent(const edm::Event&, const edm::EventSetup&);
  const float getEffectiveArea(float eta) const;
  void printEffectiveAreas() const;

  edm::EDGetTokenT<double> theRhoToken;
  edm::EDGetTokenT<reco::PhotonCollection> thePhotonToken;
  edm::Handle<double> _rhoHandle;

  std::vector<double> absEtaMin_;            // low limit of the eta range
  std::vector<double> absEtaMax_;            // upper limit of the eta range
  std::vector<double> effectiveAreaValues_;  // effective area for this eta range
  
  edm::ParameterSet phIDWP;

  vector<double> sigmaIEtaIEtaCut;
  vector<double> hOverECut;
  vector<double> relCombIso;

};

void IsoPhotonEBSelector::printEffectiveAreas() const {
  printf("  eta_min   eta_max    effective area\n");
  uint nEtaBins = absEtaMin_.size();
  for (uint iEta = 0; iEta < nEtaBins; iEta++) {
    printf("  %8.4f    %8.4f   %8.5f\n", absEtaMin_[iEta], absEtaMax_[iEta], effectiveAreaValues_[iEta]);
  }
}
const float IsoPhotonEBSelector::getEffectiveArea(float eta) const {
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

IsoPhotonEBSelector::IsoPhotonEBSelector(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : theRhoToken(iC.consumes<double>(cfg.getParameter<edm::InputTag>("rho"))) {
  absEtaMin_ = cfg.getParameter<std::vector<double> >("absEtaMin");
  absEtaMax_ = cfg.getParameter<std::vector<double> >("absEtaMax");
  effectiveAreaValues_ = cfg.getParameter<std::vector<double> >("effectiveAreaValues");
  //printEffectiveAreas();
  phIDWP = cfg.getParameter<edm::ParameterSet>("phID");

  sigmaIEtaIEtaCut = phIDWP.getParameter<std::vector<double> >("full5x5_sigmaIEtaIEtaCut");
  hOverECut = phIDWP.getParameter<std::vector<double> >("hOverECut");
  relCombIso = phIDWP.getParameter<std::vector<double> >("relCombIsolationWithEACut");
}

void IsoPhotonEBSelector::newEvent(const edm::Event& ev, const edm::EventSetup&) {
  ev.getByToken(theRhoToken, _rhoHandle);
}

bool IsoPhotonEBSelector::operator()(const reco::Photon& ph) const {
  float pt_e = ph.pt();
  unsigned int ind = 0;
  float abseta = fabs((ph.superCluster().get())->position().eta());

  if (ph.isEB()) {
    if (abseta > 1.479)
      return false;  // check if it is really needed
  }
  if (ph.isEE()) {
    ind = 1;
    if (abseta < 1.479)
      return false;  // check if it is really needed
    if (abseta >= 2.5)
      return false;  // check if it is really needed
  }

  if (ph.full5x5_sigmaIetaIeta() > sigmaIEtaIEtaCut[ind])
    return false;
  if (ph.hadronicOverEm() > hOverECut[ind])
    return false;
  const float eA = getEffectiveArea(abseta);
  const float rho = _rhoHandle.isValid() ? (float)(*_rhoHandle.product()) : 0;
  if ((ph.getPflowIsolationVariables().chargedHadronIso +
       std::max(float(0.0),
                ph.getPflowIsolationVariables().neutralHadronIso + ph.getPflowIsolationVariables().photonIso - eA * rho)) >
      relCombIso[ind] * pt_e)
    return false;

  return true;
}

EVENTSETUP_STD_INIT(IsoPhotonEBSelector);

typedef SingleObjectSelector<edm::View<reco::Photon>, IsoPhotonEBSelector> IsoPhotonEBSelectorAndSkim;

DEFINE_FWK_MODULE(IsoPhotonEBSelectorAndSkim);
