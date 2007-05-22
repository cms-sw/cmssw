#ifndef CD_NuclearSeedGenerator_H_
#define CD_NuclearSeedGenerator_H_
// -*- C++ -*-
//
// Package:    NuclearSeedGenerator
// Class:      NuclearSeedGenerator
// 
/**\class NuclearSeedGenerator NuclearSeedGenerator.h RecoTracker/NuclearSeedGenerator/interface/NuclearSeedGenerator.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincent ROBERFROID
//         Created:  Wed Feb 28 12:05:36 CET 2007
// $Id: NuclearSeedGenerator.h,v 1.2 2007/04/24 17:34:24 roberfro Exp $
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

namespace reco {class TransientTrack;}

class Trajectory;

/** \class NuclearSeedGenerator
 *
 */

class NuclearSeedGenerator : public edm::EDProducer {
   public:
      explicit NuclearSeedGenerator(const edm::ParameterSet&);
      ~NuclearSeedGenerator();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      std::auto_ptr<NuclearInteractionFinder>     theNuclearInteractionFinder;
};
#endif
