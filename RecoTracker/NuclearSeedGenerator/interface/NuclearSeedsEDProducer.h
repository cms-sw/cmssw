#ifndef CD_NuclearSeedsEDProducer_H_
#define CD_NuclearSeedsEDProducer_H_
// -*- C++ -*-
//
// Package:    NuclearSeedsEDProducer
// Class:      NuclearSeedsEDProducer
//
/**\class NuclearSeedsEDProducer NuclearSeedsEDProducer.h RecoTracker/NuclearSeedGenerator/interface/NuclearSeedsEDProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincent ROBERFROID
//         Created:  Wed Feb 28 12:05:36 CET 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionFinder.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"

namespace reco {class TransientTrack;}

class Trajectory;

/** \class NuclearSeedsEDProducer
 *
 */

class NuclearSeedsEDProducer : public edm::stream::EDProducer<> {

   public:
      explicit NuclearSeedsEDProducer(const edm::ParameterSet&);
      ~NuclearSeedsEDProducer();

   private:
      virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      std::unique_ptr<NuclearInteractionFinder>     theNuclearInteractionFinder;

      bool improveSeeds;
      edm::EDGetTokenT<TrajectoryCollection> producer_;
      edm::EDGetTokenT<MeasurementTrackerEvent> mteToken_;
};
#endif
