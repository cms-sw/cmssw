#ifndef GoodSeedProducer_H
#define GoodSeedProducer_H
// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
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
      virtual void beginRun(const edm::Run & run,const edm::EventSetup&) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endRun(const edm::Run & run,const edm::EventSetup&) override;
 
      ///Find the bin in pt and eta
      int getBin(float,float);
      int getBin(float);
      void PSforTMVA(const math::XYZTLorentzVector& mom,
		     const math::XYZTLorentzVector& pos);
      bool IsIsolated(float  charge,float P,
	              math::XYZPointF, 
                      const reco::PFClusterCollection &ecalColl,
                      const reco::PFClusterCollection &hcalColl);

      void fillPreIdRefValueMap( edm::Handle<reco::TrackCollection> tkhandle,
				 const edm::OrphanHandle<reco::PreIdCollection>&,
				 edm::ValueMap<reco::PreIdRef>::Filler & filler);
      // ----------member data ---------------------------

      ///Vector of clusters of the PreShower
      std::vector<reco::PFCluster> ps1Clus;
      std::vector<reco::PFCluster> ps2Clus;

      ///Name of the Seed(Ckf) Collection
      std::string preidckf_;

      ///Name of the Seed(Gsf) Collection
      std::string preidgsf_;

      ///Name of the preid Collection (FB)
      std::string preidname_;

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
      
      ///ISOLATION REQUEST AS DONE IN THE TAU GROUP
      bool applyIsolation_;
      double HcalIsolWindow_;
      double EcalStripSumE_minClusEnergy_;
      double EcalStripSumE_deltaEta_;
      double EcalStripSumE_deltaPhiOverQ_minValue_;
      double EcalStripSumE_deltaPhiOverQ_maxValue_;
      double minEoverP_;
      double maxHoverP_;
      ///

      ///Cut on the energy of the clusters
      double clusThreshold_;

      ///Min and MAx allowed values forEoverP
      double minEp_;
      double maxEp_;

      ///Produce the Seed for Ckf tracks?
      bool produceCkfseed_;

      ///  switch to disable the pre-id
      bool disablePreId_;

      ///Produce the pre-id debugging collection 
      bool producePreId_;
      
      /// Threshold to save Pre Idinfo
      double PtThresholdSavePredId_;

      ///vector of thresholds for different bins of eta and pt
      float thr[150];
      float thrPS[20];

      // ----------access to event data
      edm::ParameterSet conf_;
      edm::InputTag pfCLusTagPSLabel_;
      edm::InputTag pfCLusTagECLabel_;
      edm::InputTag pfCLusTagHCLabel_;
      std::vector<edm::InputTag> tracksContainers_;
      

      std::string fitterName_;
      std::string smootherName_;
      std::string propagatorName_;

      PFResolutionMap* resMapEtaECAL_;
      PFResolutionMap* resMapPhiECAL_;

      ///TRACK QUALITY
      bool useQuality_;
      reco::TrackBase::TrackQuality trackQuality_;
	
      ///READER FOR TMVA
      TMVA::Reader *reader;

      ///VARIABLES NEEDED FOR TMVA
      float eP,chi,eta,pt,nhit,dpt,chired,chiRatio;
      float ps1En,ps2En,ps1chi,ps2chi;
      ///USE OF TMVA 
      bool useTmva_;

      ///TMVA method
      std::string method_;

      ///B field
      math::XYZVector B_;

      ///Use of Preshower clusters
      bool usePreshower_;

      /// Map used to create the TrackRef, PreIdRef value map
      std::map<reco::TrackRef,unsigned> refMap_;
     
};
#endif
