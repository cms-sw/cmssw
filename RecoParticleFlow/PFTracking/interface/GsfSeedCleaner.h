#ifndef GsfSeedCleaner_H
#define GsfSeedCleaner_H
// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class GsfSeedCleaner : public edm::EDProducer {

   public:
      explicit GsfSeedCleaner(const edm::ParameterSet&);
      ~GsfSeedCleaner(){};
  
   private:
      virtual void beginJob(const edm::EventSetup&){} ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob(){} ;
      bool CompareHits(const reco::GsfTrack tk,const TrajectorySeed s);

      // ----------access to event data
      edm::ParameterSet conf_;
      edm::InputTag preIdLabel_;
      std::vector<edm::InputTag> tracksContainers_;
      
     
      
};
#endif
