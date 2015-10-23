#ifndef RecoMuon_GlobalTrackingTools_GlobalMuonRefitter_H
#define RecoMuon_GlobalTrackingTools_GlobalMuonRefitter_H

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {class Event; class EventSetup;}
namespace reco {class TransientTrack;}

class MuonServiceProxy;

class MuonSegmentMatcher { 

  public:

    /// constructor with Parameter Set and MuonServiceProxy
  MuonSegmentMatcher(const edm::ParameterSet&, edm::ConsumesCollector& iC);
          
    /// destructor
    virtual ~MuonSegmentMatcher();

    /// perform the matching
    std::vector<const DTRecSegment4D*> matchDT (const reco::Track& muon, const edm::Event& event);
   
    std::vector<const CSCSegment*>     matchCSC(const reco::Track& muon, const edm::Event& event);

    std::vector<const RPCRecHit*>     matchRPC(const reco::Track& muon, const edm::Event& event);
  
  protected:

  private:
	const MuonServiceProxy* theService;
	const edm::Event* theEvent;
	
	edm::InputTag TKtrackTags_;
	edm::InputTag trackTags_; //used to select what tracks to read from configuration file
	edm::InputTag DTSegmentTags_;
	edm::InputTag CSCSegmentTags_;
	edm::InputTag RPCHitTags_;

	edm::EDGetTokenT<DTRecSegment4DCollection> dtRecHitsToken;
	edm::EDGetTokenT<CSCSegmentCollection> allSegmentsCSCToken;
	edm::EDGetTokenT<RPCRecHitCollection> rpcRecHitsToken;

	double dtRadius_;
	
	bool dtTightMatch;
	bool cscTightMatch;

};
#endif
