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
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TMVA/Reader.h"
#include "DataFormats/Math/interface/LorentzVector.h"
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

//namespace reco {
class PFResolutionMap;
// }

class PFTrackTransformer;
class TrajectoryFitter;
class TrajectorySmoother;
class TrackerGeometry;
class TrajectoryStateOnSurface;

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
     int getBin(float);
      void PSforTMVA(math::XYZTLorentzVector mom,
		     math::XYZTLorentzVector pos);
      // ----------member data ---------------------------

      ///Vector of clusters of the PreShower
      std::vector<reco::PFCluster> ps1Clus;
      std::vector<reco::PFCluster> ps2Clus;

      ///Name of the Seed(Ckf) Collection
      std::string preidckf_;

      ///Name of the Seed(Gsf) Collection
      std::string preidgsf_;

      ///Fitter
      edm::ESHandle<TrajectoryFitter> fitter_;

      ///Smoother
      edm::ESHandle<TrajectorySmoother> smoother_;

      ///PFTrackTransformer
      PFTrackTransformer *pfTransformer_;

      ///Number of hits in the seed;
      int nHitsInSeed_;

      ///Minimum transverse momentum and maximum pseudorapidity
      double minPt_;
      double maxPt_;
      double maxEta_;

      ///Cut on the energy of the clusters
      double clusThreshold_;

      ///Min and MAx allowed values forEoverP
      double minEp_;
      double maxEp_;

      ///Produce the Seed for Ckf tracks?
      bool produceCkfseed_;

      ///Produce the PFtracks for Ckf tracks? 
      bool produceCkfPFT_;

      ///vector of thresholds for different bins of eta and pt
      float thr[150];
      float thrPS[20];

      // ----------access to event data
      edm::ParameterSet conf_;
      edm::InputTag pfCLusTagPSLabel_;
      edm::InputTag pfCLusTagECLabel_;
      std::vector<edm::InputTag> tracksContainers_;

      std::string fitterName_;
      std::string smootherName_;
      std::string propagatorName_;

      static PFResolutionMap* resMapEtaECAL_;
      static PFResolutionMap* resMapPhiECAL_;

      ///READER FOR TMVA
      TMVA::Reader *reader;

      ///VARIABLES NEEDED FOR TMVA
      float eP,chi,eta,pt,nhit,dpt,chired,chiRatio;
      float ps1En,ps2En,ps1chi,ps2chi;
      ///USE OF TMVA 
      bool useTmva_;

      ///TMVA method
      std::string method_;
};
#endif
