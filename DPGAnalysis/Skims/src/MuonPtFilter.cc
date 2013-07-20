/** \file
 *
 * $Date: 2012/01/21 17:15:54 $
 * $Revision: 1.4 $
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
MuonPtFilter::MuonPtFilter(const edm::ParameterSet& pset)
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
bool MuonPtFilter::filter(edm::Event& event, const edm::EventSetup& eventSetup) {
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);
  
  reco::TrackCollection::const_iterator staTrack;
  
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    if (staTrack->pt() > theMinPt)
      return true;
  }

  return false;
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonPtFilter);
