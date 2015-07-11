#ifndef RecoParticleFlow_PFTracking_GoodSeedProducer_H
#define RecoParticleFlow_PFTracking_GoodSeedProducer_H
// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoParticleFlow/PFTracking/interface/PFGeometry.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

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



namespace goodseedhelpers {
  class HeavyObjectCache {
    constexpr static unsigned int kMaxWeights = 9;
  public:
    HeavyObjectCache(const edm::ParameterSet& conf);
    std::array<std::unique_ptr<const GBRForest>,kMaxWeights> gbr;    
  private:
    // for temporary variable binding while reading
    float eP,eta,pt,nhit,dpt,chired,chiRatio;
    float chikfred,trk_ecalDeta,trk_ecalDphi;    
  };
}

class GoodSeedProducer final : public edm::stream::EDProducer<edm::GlobalCache<goodseedhelpers::HeavyObjectCache> > {
  typedef TrajectoryStateOnSurface TSOS;
 public:
  explicit GoodSeedProducer(const edm::ParameterSet&, const goodseedhelpers::HeavyObjectCache*);
  
  static std::unique_ptr<goodseedhelpers::HeavyObjectCache> 
    initializeGlobalCache( const edm::ParameterSet& conf ) {
       return std::unique_ptr<goodseedhelpers::HeavyObjectCache>(new goodseedhelpers::HeavyObjectCache(conf));
   }
  
  static void globalEndJob(goodseedhelpers::HeavyObjectCache const* ) {
  }

   private:
      virtual void beginRun(const edm::Run & run,const edm::EventSetup&) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
 
      ///Find the bin in pt and eta
      int getBin(float,float);

      void fillPreIdRefValueMap( edm::Handle<reco::TrackCollection> tkhandle,
				 const edm::OrphanHandle<reco::PreIdCollection>&,
				 edm::ValueMap<reco::PreIdRef>::Filler & filler);
      // ----------member data ---------------------------

      ///Name of the Seed(Ckf) Collection
      std::string preidckf_;

      ///Name of the Seed(Gsf) Collection
      std::string preidgsf_;

      ///Name of the preid Collection (FB)
      std::string preidname_;

      ///Fitter
      std::unique_ptr<TrajectoryFitter> fitter_;

      ///Smoother
      std::unique_ptr<TrajectorySmoother> smoother_;

      // needed by the above
      TkClonerImpl hitCloner;

      ///PFTrackTransformer
      std::unique_ptr<PFTrackTransformer> pfTransformer_;

      ///Number of hits in the seed;
      int nHitsInSeed_;

      ///Minimum transverse momentum and maximum pseudorapidity
      double minPt_;
      double maxPt_;
      double maxEta_;
      
      double HcalIsolWindow_;
      double EcalStripSumE_minClusEnergy_;
      double EcalStripSumE_deltaEta_;
      double EcalStripSumE_deltaPhiOverQ_minValue_;
      double EcalStripSumE_deltaPhiOverQ_maxValue_;
      double minEoverP_;
      double maxHoverP_;

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

      // ----------access to event data
      edm::ParameterSet conf_;
      edm::EDGetTokenT<reco::PFClusterCollection> pfCLusTagPSLabel_;
      edm::EDGetTokenT<reco::PFClusterCollection> pfCLusTagECLabel_;
      edm::EDGetTokenT<reco::PFClusterCollection> pfCLusTagHCLabel_;
      std::vector<edm::EDGetTokenT<std::vector<Trajectory> > > trajContainers_;
      std::vector<edm::EDGetTokenT<reco::TrackCollection > > tracksContainers_;
      
      std::string fitterName_;
      std::string smootherName_;
      std::string propagatorName_;
      std::string trackerRecHitBuilderName_;

      std::unique_ptr<PFResolutionMap> resMapEtaECAL_;
      std::unique_ptr<PFResolutionMap> resMapPhiECAL_;

      ///TRACK QUALITY
      bool useQuality_;
      reco::TrackBase::TrackQuality trackQuality_;
      
      ///VARIABLES NEEDED FOR TMVA
      float eP,eta,pt,nhit,dpt,chired,chiRatio;
      float chikfred,trk_ecalDeta,trk_ecalDphi;                      
      double Min_dr_;

      ///USE OF TMVA 
      bool useTmva_;

      ///TMVA method
      std::string method_;

      ///B field
      math::XYZVector B_;

      /// Map used to create the TrackRef, PreIdRef value map
      std::map<reco::TrackRef,unsigned> refMap_;

      PFGeometry pfGeometry_;
};
#endif
