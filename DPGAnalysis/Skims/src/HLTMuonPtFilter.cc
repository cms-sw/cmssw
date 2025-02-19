/** \file
 *
 * $Date: 2012/01/21 18:00:01 $
 * $Revision: 1.4 $
 * \author Silvia Goy Lopez - CERN <silvia.goy.lopez@cern.ch>
 */

/* This Class Header */
#include "DPGAnalysis/Skims/interface/HLTMuonPtFilter.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

/* C++ Headers */
using namespace std;
using namespace edm;

/* ====================================================================== */

/// Constructor
HLTMuonPtFilter::HLTMuonPtFilter(const edm::ParameterSet& pset) :
  HLTFilter(pset)
{
  // the name of the STA rec hits collection
  theSTAMuonLabel = pset.getParameter<std::string>("SALabel");

  theMinPt = pset.getParameter<double>("minPt"); // pt min (GeV)

  LogDebug("HLTMuonPt") << " SALabel : " << theSTAMuonLabel 
    << " Min Pt : " << theMinPt;
}

/// Destructor
HLTMuonPtFilter::~HLTMuonPtFilter() {
}

/* Operations */ 
bool HLTMuonPtFilter::hltFilter(edm::Event& event, const edm::EventSetup& eventSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) {
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);
  
  reco::TrackCollection::const_iterator staTrack;
  
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack) {
    if (staTrack->pt()>theMinPt)
      return true;
  }

  return false;
}

// define this as a plug-in
DEFINE_FWK_MODULE(HLTMuonPtFilter);
