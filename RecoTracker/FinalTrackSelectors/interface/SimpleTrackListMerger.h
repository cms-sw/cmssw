#ifndef SimpleTrackListMerger_h
#define SimpleTrackListMerger_h

//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           SimpleTrackListMerger
// 
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: muzaffar $
// $Date: 2013/02/27 13:28:30 $
// $Revision: 1.8 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace cms
{
  class SimpleTrackListMerger : public edm::EDProducer
  {
  public:

    explicit SimpleTrackListMerger(const edm::ParameterSet& conf);

    virtual ~SimpleTrackListMerger();

    virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ParameterSet conf_;

    std::auto_ptr<reco::TrackCollection> outputTrks;
    std::auto_ptr<reco::TrackExtraCollection> outputTrkExtras;
    std::auto_ptr< TrackingRecHitCollection>  outputTrkHits;
    std::auto_ptr< std::vector<Trajectory> > outputTrajs;
    std::auto_ptr< TrajTrackAssociationCollection >  outputTTAss;
    std::auto_ptr< TrajectorySeedCollection > outputSeeds;

    reco::TrackRefProd refTrks;
    reco::TrackExtraRefProd refTrkExtras;
    TrackingRecHitRefProd refTrkHits;
    edm::RefProd< std::vector<Trajectory> > refTrajs;
    std::vector<reco::TrackRef> trackRefs;
    edm::RefProd< TrajectorySeedCollection > refTrajSeeds;
    
    bool copyExtras_;
    bool makeReKeyedSeeds_;
  };
}


#endif
