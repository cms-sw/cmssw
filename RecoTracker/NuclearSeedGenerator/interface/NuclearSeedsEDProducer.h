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
// $Id: NuclearSeedsEDProducer.h,v 1.5 2007/10/09 14:55:03 roberfro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearInteractionFinder.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"

#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"

namespace reco {class TransientTrack;}

class Trajectory;

/** \class NuclearSeedsEDProducer
 *
 */

class NuclearSeedsEDProducer : public edm::EDProducer {

   public:
      explicit NuclearSeedsEDProducer(const edm::ParameterSet&);
      ~NuclearSeedsEDProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      std::auto_ptr<NuclearInteractionFinder>     theNuclearInteractionFinder;

      bool improveSeeds;
      std::string producer_;
      std::string   navigationSchoolName;
      const NavigationSchool*         theNavigationSchool;
};
#endif
