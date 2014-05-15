#ifndef ConversionTrackProducer_h
#define ConversionTrackProducer_h

//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackProducer
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

#include "DataFormats/EgammaTrackReco/interface/ConversionTrack.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"

namespace reco {
  class BeamSpot;
}

//--------------------------------------------------
//Added by D. Giordano
// 2011/08/05
// Reduction of the track sample based on geometric hypothesis for conversion tracks
#include "RecoTracker/ConversionSeedGenerators/interface/IdealHelixParameters.h"
//--------------------------------------------------

  class ConversionTrackProducer : public edm::stream::EDProducer<>
  {

    typedef edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>,
      reco::GsfTrackCollection,unsigned short> > 
      TrajGsfTrackAssociationCollection;

  public:

    explicit ConversionTrackProducer(const edm::ParameterSet& conf);

    virtual ~ConversionTrackProducer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:

    edm::ParameterSet conf_;

    std::string trackProducer;
    edm::EDGetTokenT<edm::View<reco::Track> > genericTracks ;
    edm::EDGetTokenT<TrajTrackAssociationCollection> kfTrajectories; 
    edm::EDGetTokenT<TrajGsfTrackAssociationCollection> gsfTrajectories;
    bool useTrajectory;
    bool setTrackerOnly;
    bool setArbitratedEcalSeeded;
    bool setArbitratedMerged;
    bool setArbitratedMergedEcalGeneral;

    //--------------------------------------------------
    //Added by D. Giordano
    // 2011/08/05
    // Reduction of the track sample based on geometric hypothesis for conversion tracks

    edm::EDGetTokenT<reco::BeamSpot> beamSpotInputTag;
    bool filterOnConvTrackHyp;
    double minConvRadius;
    IdealHelixParameters ConvTrackPreSelector;
    //--------------------------------------------------

    std::auto_ptr<reco::ConversionTrackCollection> outputTrks;
  };
#endif
