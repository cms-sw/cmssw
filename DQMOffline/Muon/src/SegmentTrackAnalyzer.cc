
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/28 15:21:03 $
 *  $Revision: 1.6 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/SegmentTrackAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



SegmentTrackAnalyzer::SegmentTrackAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {

  cout<<"[SegmentTrackAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

  const ParameterSet SegmentsTrackAssociatorParameters = parameters.getParameter<ParameterSet>("SegmentsTrackAssociatorParameters");
  theSegmentsAssociator = new SegmentsTrackAssociator(SegmentsTrackAssociatorParameters);

}


SegmentTrackAnalyzer::~SegmentTrackAnalyzer() { }


void SegmentTrackAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "segmTrackAnalyzer";
  LogTrace(metname)<<"[SegmentTrackAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/SegmentTrackAnalyzer");

}


void SegmentTrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& recoTrack){

  LogTrace(metname)<<"[SegmentTrackAnalyzer] Filling the histos";
  
  MuonTransientTrackingRecHit::MuonRecHitContainer TheSegments = theSegmentsAssociator->associate(iEvent, iSetup, recoTrack );

  LogTrace(metname)<<"[SegmentTrackAnalyzer] # of segments associated to the track: "<<(TheSegments).size();

}
