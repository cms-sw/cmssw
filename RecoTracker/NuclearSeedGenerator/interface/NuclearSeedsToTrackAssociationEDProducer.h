#ifndef CD_NuclearSeedsToTrackAssociationEDProducer_H_
#define CD_NuclearSeedsToTrackAssociationEDProducer_H_
// -*- C++ -*-
//
// Package:    NuclearAssociatonMapEDProducer
// Class:      NuclearSeedsToTrackAssociationEDProducer
//
/**\class NuclearSeedsToTrackAssociationEDProducer NuclearSeedsToTrackAssociationEDProducer.h RecoTracker/NuclearSeedGenerator/interface/NuclearSeedsToTrackAssociationEDProducer.h

 Description: Associate nuclear seeds to primary tracks and associate secondary tracks to primary tracks

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincent ROBERFROID
//         Created:  Fri Aug 10 12:05:36 CET 2007
// $Id: NuclearSeedsToTrackAssociationEDProducer.h,v 1.2 2007/08/10 09:12:11 roberfro Exp $
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

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"

class NuclearSeedsToTrackAssociationEDProducer : public edm::EDProducer {

public:
      explicit NuclearSeedsToTrackAssociationEDProducer(const edm::ParameterSet&);
      ~NuclearSeedsToTrackAssociationEDProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

      // ----------member data ---------------------------
      edm::ParameterSet conf_;
};
#endif
