/** \class RecoDiMuon
 *
 *  
 *  This class is an EDFilter choosing reconstructed di-muons
 *
 *
 *  \author Chang Liu  -  Purdue University
 *
 */

//-ap #include "Configuration/CSA06Skimming/interface/RecoDiMuon.h"
#include "GeneratorInterface/GenFilters/plugins/RecoDiMuon.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

RecoDiMuon::RecoDiMuon(const edm::ParameterSet& iConfig) : nEvents_(0), nAccepted_(0) {
  muonLabel_ = iConfig.getParameter<InputTag>("MuonLabel");
  singleMuonPtMin_ = iConfig.getUntrackedParameter<double>("SingleMuonPtMin", 20.);
  diMuonPtMin_ = iConfig.getUntrackedParameter<double>("DiMuonPtMin", 5.);
}

RecoDiMuon::~RecoDiMuon() {}

void RecoDiMuon::endJob() {
  edm::LogVerbatim("RecoDiMuon") << "Events read " << nEvents_ << " Events accepted " << nAccepted_ << "\nEfficiency "
                                 << ((double)nAccepted_) / ((double)nEvents_) << std::endl;
}

// ------------ method called to skim the data  ------------
bool RecoDiMuon::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nEvents_++;
  bool accepted = false;
  using namespace edm;

  Handle<reco::TrackCollection> muons;

  iEvent.getByLabel(muonLabel_, muons);
  if (!muons.isValid()) {
    edm::LogError("RecoDiMuon") << "FAILED to get Muon Track Collection. ";
    return false;
  }

  if (muons->empty()) {
    return false;
  }

  // at least one muons above a pt threshold singleMuonPtMin
  // or at least 2 muons above a pt threshold diMuonPtMin
  int nMuonOver2ndCut = 0;
  for (reco::TrackCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
    if (muon->pt() > singleMuonPtMin_)
      accepted = true;
    if (muon->pt() > diMuonPtMin_)
      nMuonOver2ndCut++;
  }
  if (nMuonOver2ndCut >= 2)
    accepted = true;

  if (accepted)
    nAccepted_++;

  return accepted;
}
