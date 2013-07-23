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
// $Author: giordano $
// $Date: 2011/08/05 19:45:49 $
// $Revision: 1.4 $
//

#include "FWCore/Framework/interface/EDProducer.h"
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


//--------------------------------------------------
//Added by D. Giordano
// 2011/08/05
// Reduction of the track sample based on geometric hypothesis for conversion tracks
#include "RecoTracker/ConversionSeedGenerators/interface/IdealHelixParameters.h"
//--------------------------------------------------

  class ConversionTrackProducer : public edm::EDProducer
  {

  public:

    explicit ConversionTrackProducer(const edm::ParameterSet& conf);

    virtual ~ConversionTrackProducer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:

    edm::ParameterSet conf_;

    std::string trackProducer;
    bool useTrajectory;
    bool setTrackerOnly;
    bool setArbitratedEcalSeeded;
    bool setArbitratedMerged;
    bool setArbitratedMergedEcalGeneral;

    //--------------------------------------------------
    //Added by D. Giordano
    // 2011/08/05
    // Reduction of the track sample based on geometric hypothesis for conversion tracks

    edm::InputTag beamSpotInputTag;
    bool filterOnConvTrackHyp;
    double minConvRadius;
    IdealHelixParameters ConvTrackPreSelector;
    //--------------------------------------------------

    std::auto_ptr<reco::ConversionTrackCollection> outputTrks;
  };
#endif
