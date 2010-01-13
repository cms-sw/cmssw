#ifndef Calibration_EcalIsolatedParticleCandidateProducer_h
#define Calibration_EcalIsolatedParticleCandidateProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
//
// class decleration
//

class EcalIsolatedParticleCandidateProducer : public edm::EDProducer {
   public:
      explicit EcalIsolatedParticleCandidateProducer(const edm::ParameterSet&);
      ~EcalIsolatedParticleCandidateProducer();

   private:

    const CaloGeometry* geo;

    double InConeSize_;
    double OutConeSize_;
    double hitCountEthr_;
    double hitEthr_;
    edm::InputTag l1tausource_;
    edm::InputTag hltGTseedlabel_;
    edm::InputTag EBrecHitCollectionLabel_;
    edm::InputTag EErecHitCollectionLabel_;

      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
};

#endif
