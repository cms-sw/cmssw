#ifndef Calibration_EcalIsolatedParticleCandidateProducer_h
#define Calibration_EcalIsolatedParticleCandidateProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidate.h"
//
// class decleration
//

class EcalIsolatedParticleCandidateProducer : public edm::EDProducer {
   public:
      explicit EcalIsolatedParticleCandidateProducer(const edm::ParameterSet&);
      ~EcalIsolatedParticleCandidateProducer();

   private:

      bool useEndcap_;
      double coneSize_;
      double minEnergy_;
      std::string barrelBclusterProducer_;
      std::string endcapBclusterProducer_;
      std::string barrelBclusterCollectionLabel_;
      std::string endcapBclusterCollectionLabel_;

      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
};

#endif
