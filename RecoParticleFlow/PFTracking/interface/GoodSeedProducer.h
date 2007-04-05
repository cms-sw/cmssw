#ifndef GoodSeedProducer_H
#define GoodSeedProducer_H
// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/// \brief Abstract
/*!
\author Michele Pioppi
\date January 2007

 GoodSeedProducer is the base class
 for electron preidentification in PFLow FW.
 It reads refitted tracks and PFCluster collection, 
 and following some criteria divides electrons from hadrons.
 Then it saves the seed of the tracks preidentified as electrons.
 It also transform  all the tracks in the first PFRecTrack collection.
*/

namespace reco {
  class PFResolutionMap;
}

class PFTrackTransformer;
class TrajectoryFitter;
class TrajectorySmoother;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class Propagator;

class GoodSeedProducer : public edm::EDProducer {
  typedef TrajectoryStateOnSurface TSOS;
   public:
      explicit GoodSeedProducer(const edm::ParameterSet&);
      ~GoodSeedProducer();
  
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob(){}
 
      ///Find the bin in pt and eta
      int getBin(float,float);

      // ----------member data ---------------------------

      ///Name of the Seed(Ckf) Collection
      std::string preidckf_;

      ///Name of the Seed(Gsf) Collection
      std::string preidgsf_;

      ///Propagator
      edm::ESHandle<Propagator> propagator_;

      ///Fitter
      edm::ESHandle<TrajectoryFitter> fitter_;

      ///Smoother
      edm::ESHandle<TrajectorySmoother> smoother_;

      ///PFTrackTransformer
      PFTrackTransformer *pfTransformer_;

      ///Number of hits in the seed;
      int nHitsInSeed_;

      ///Cut on the energy of the clusters
      double clusThreshold_;

      ///Produce the Seed for Ckf tracks?
      bool produceCkfseed_;

      ///Produce the PFtracks for Ckf tracks? 
      bool produceCkfPFT_;

      ///vector of thresholds for different bins of eta and pt
      float thr[150];

      // ----------access to event data
      edm::ParameterSet conf_;
      edm::InputTag pfCLusTagPSLabel_;
      edm::InputTag pfCLusTagECLabel_;
      edm::InputTag refitLabel_;

      std::string fitterName_;
      std::string smootherName_;
      std::string propagatorName_;

      static reco::PFResolutionMap* resMapEtaECAL_;
      static reco::PFResolutionMap* resMapPhiECAL_;
};
#endif
