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
//

#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include "DataFormats/Common/interface/ValueMap.h"

namespace cms
{
  class TrackListMerger : public edm::stream::EDProducer<>
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

    struct TkEDGetTokenss  {
        edm::InputTag tag;
        edm::EDGetTokenT<reco::TrackCollection> tk;
        edm::EDGetTokenT<std::vector<Trajectory> >        traj;
        edm::EDGetTokenT<TrajTrackAssociationCollection > tass;
        edm::EDGetTokenT<edm::ValueMap<int>   > tsel;
        edm::EDGetTokenT<edm::ValueMap<float> > tmva;
        TkEDGetTokenss() {}
        TkEDGetTokenss(const edm::InputTag &tag_, edm::EDGetTokenT<reco::TrackCollection> && tk_, 
                 edm::EDGetTokenT<std::vector<Trajectory> > && traj_, edm::EDGetTokenT<TrajTrackAssociationCollection > && tass_,
                 edm::EDGetTokenT<edm::ValueMap<int> > &&tsel_, edm::EDGetTokenT<edm::ValueMap<float> > && tmva_) :
            tag(tag_), tk(tk_), traj(traj_), tass(tass_), tsel(tsel_), tmva(tmva_) {}
    };
    TkEDGetTokenss edTokens(const edm::InputTag &tag, const edm::InputTag &seltag, const edm::InputTag &mvatag) {
        return TkEDGetTokenss(tag, consumes<reco::TrackCollection>(tag), 
                                consumes<std::vector<Trajectory> >(tag), consumes<TrajTrackAssociationCollection >(tag),
                                consumes<edm::ValueMap<int> >(seltag), consumes<edm::ValueMap<float> >(mvatag));
    }
    TkEDGetTokenss edTokens(const edm::InputTag &tag, const edm::InputTag &mvatag) {
        return TkEDGetTokenss(tag, consumes<reco::TrackCollection>(tag), 
                                consumes<std::vector<Trajectory> >(tag), consumes<TrajTrackAssociationCollection >(tag),
                                edm::EDGetTokenT<edm::ValueMap<int> >(), consumes<edm::ValueMap<float> >(mvatag));
    }
    std::vector<TkEDGetTokenss>      trackProducers_;

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

    bool allowFirstHitShare_;
    reco::TrackBase::TrackQuality qualityToSet_;
    bool use_sharesInput_;
    bool trkQualMod_;

  };
}


#endif
