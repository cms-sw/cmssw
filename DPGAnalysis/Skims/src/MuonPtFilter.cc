/** \file
 *
 * $Date: 2010/08/07 14:55:55 $
 * $Revision: 1.2 $
 * \author Silvia Goy Lopez - CERN <silvia.goy.lopez@cern.ch>
 */

/* This Class Header */
#include "DPGAnalysis/Skims/interface/MuonPtFilter.h"

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
MuonPtFilter::MuonPtFilter(const edm::ParameterSet& pset) :
    HLTFilter(pset)
{
  // the name of the STA rec hits collection
  theSTAMuonLabel = pset.getParameter<std::string>("SALabel");

  theMinPt = pset.getParameter<double>("minPt"); // pt min (GeV)

  LogDebug("MuonPt") << " SALabel : " << theSTAMuonLabel 
    << " Min Pt : " << theMinPt;
}

/// Destructor
MuonPtFilter::~MuonPtFilter() {
}

/* Operations */ 
bool MuonPtFilter::hltFilter(edm::Event& event, const edm::EventSetup& eventSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) {
  bool accept = false;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);
  
  reco::TrackCollection::const_iterator staTrack;
  
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    
    if(staTrack->pt()>theMinPt){
      accept=true;
      return accept;
    }

  }

  return accept;


}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonPtFilter);
