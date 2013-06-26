#ifndef TrackListMerger_h
#define TrackListMerger_h

//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           TrackListMerger
// 
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: mwalker $
// $Date: 2013/04/16 16:23:54 $
// $Revision: 1.11 $
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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace cms
{
  class TrackListMerger : public edm::EDProducer
  {
  public:

    explicit TrackListMerger(const edm::ParameterSet& conf);

    virtual ~TrackListMerger();

    virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

  private:
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
    edm::RefProd< TrajectorySeedCollection > refTrajSeeds;
    
    bool copyExtras_;
    bool makeReKeyedSeeds_;

    std::vector<edm::InputTag> trackProducers_;
    double maxNormalizedChisq_;
    double minPT_;
    unsigned int minFound_;
    float epsilon_;
    float shareFrac_;
    float foundHitBonus_;
    float lostHitPenalty_;
    std::vector<double> indivShareFrac_;

    std::vector< std::vector< int> > listsToMerge_;
    std::vector<bool> promoteQuality_;
    std::vector<int> hasSelector_;
    std::vector<edm::InputTag> selectors_;
    std::vector<edm::InputTag> mvaStores_;

    bool allowFirstHitShare_;
    reco::TrackBase::TrackQuality qualityToSet_;
    bool use_sharesInput_;
    bool trkQualMod_;

  };
}


#endif
