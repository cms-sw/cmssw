#ifndef PFElecTkProducer_H
#define PFElecTkProducer_H
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"

class PFTrackTransformer;
class GsfTrack;
class MagneticField;
class TrackerGeometry;
class ConvBremPFTrackFinder;

/// \brief Abstract
/*!
\author Michele Pioppi, Daniele Benedetti
\date January 2007

 PFElecTkProducer reads the merged GsfTracks collection
 built with the TrackerDriven and EcalDriven seeds 
 and transform them in PFGsfRecTracks.
*/

class PFElecTkProducer : public edm::stream::EDProducer<> {
 public:
  
     ///Constructor
     explicit PFElecTkProducer(const edm::ParameterSet&);

     ///Destructor
     ~PFElecTkProducer();

   private:
      virtual void beginRun(const edm::Run&,const edm::EventSetup&) override;
      virtual void endRun(const edm::Run&,const edm::EventSetup&) override;

      ///Produce the PFRecTrack collection
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    
      int FindPfRef(const reco::PFRecTrackCollection & PfRTkColl, 
		    const reco::GsfTrack&, bool);
      
      bool isFifthStep(reco::PFRecTrackRef pfKfTrack);

      bool applySelection(const reco::GsfTrack&);
      
      bool resolveGsfTracks(const std::vector<reco::GsfPFRecTrack> &GsfPFVec,
			    unsigned int ngsf,
			    std::vector<unsigned int> &secondaries,
			    const reco::PFClusterCollection & theEClus);

      float minTangDist(const reco::GsfPFRecTrack& primGsf,
			const reco::GsfPFRecTrack& secGsf); 
      
      bool isSameEgSC(const reco::ElectronSeedRef& nSeedRef,
		      const reco::ElectronSeedRef& iSeedRef,
		      bool& bothGsfEcalDriven,
		      float& SCEnergy);

      bool isSharingEcalEnergyWithEgSC(const reco::GsfPFRecTrack& nGsfPFRecTrack,
				       const reco::GsfPFRecTrack& iGsfPFRecTrack,
				       const reco::ElectronSeedRef& nSeedRef,
				       const reco::ElectronSeedRef& iSeedRef,
				       const reco::PFClusterCollection& theEClus,
				       bool& bothGsfTrackerDriven,
				       bool& nEcalDriven,
				       bool& iEcalDriven,
				       float& nEnergy,
				       float& iEnergy);
      
      bool isInnerMost(const reco::GsfTrackRef& nGsfTrack,
		       const reco::GsfTrackRef& iGsfTrack,
		       bool& sameLayer);
      
      bool isInnerMostWithLostHits(const reco::GsfTrackRef& nGsfTrack,
				   const reco::GsfTrackRef& iGsfTrack,
				   bool& sameLayer);
      
      void createGsfPFRecTrackRef(const edm::OrphanHandle<reco::GsfPFRecTrackCollection>& gsfPfHandle,
				  std::vector<reco::GsfPFRecTrack>& gsfPFRecTrackPrimary,
				  const std::map<unsigned int, std::vector<reco::GsfPFRecTrack> >& MapPrimSec);
	
      // ----------member data ---------------------------
      reco::GsfPFRecTrack pftrack_;
      reco::GsfPFRecTrack secpftrack_;
      edm::ParameterSet conf_;
      edm::EDGetTokenT<reco::GsfTrackCollection> gsfTrackLabel_;
      edm::EDGetTokenT<reco::PFRecTrackCollection> pfTrackLabel_;
      edm::EDGetTokenT<reco::VertexCollection> primVtxLabel_;
      edm::EDGetTokenT<reco::PFClusterCollection> pfEcalClusters_;
      edm::EDGetTokenT<reco::PFDisplacedTrackerVertexCollection>  pfNuclear_;
      edm::EDGetTokenT<reco::PFConversionCollection> pfConv_;
      edm::EDGetTokenT<reco::PFV0Collection>  pfV0_;
      bool useNuclear_;
      bool useConversions_;
      bool useV0_;
      bool applyAngularGsfClean_;
      double detaCutGsfClean_;
      double dphiCutGsfClean_;

      ///PFTrackTransformer
      PFTrackTransformer *pfTransformer_;     
      const MultiTrajectoryStateMode *mtsMode_;
      MultiTrajectoryStateTransform  mtsTransform_;
      ConvBremPFTrackFinder *convBremFinder_;


      ///Trajectory of GSfTracks in the event?
      bool trajinev_;
      bool modemomentum_;
      bool applySel_;
      bool applyGsfClean_;
      bool useFifthStep_;
      bool useFifthStepForEcalDriven_;
      bool useFifthStepForTrackDriven_;
      //   bool useFifthStepSec_;
      bool debugGsfCleaning_;
      double SCEne_;
      double detaGsfSC_;
      double dphiGsfSC_;
      double maxPtConvReco_;
      
      /// Conv Brem Finder
      bool useConvBremFinder_;

      double mvaConvBremFinderIDBarrelLowPt_;
      double mvaConvBremFinderIDBarrelHighPt_;
      double mvaConvBremFinderIDEndcapsLowPt_;
      double mvaConvBremFinderIDEndcapsHighPt_;
      std::string path_mvaWeightFileConvBremBarrelLowPt_;
      std::string path_mvaWeightFileConvBremBarrelHighPt_;
      std::string path_mvaWeightFileConvBremEndcapsLowPt_;
      std::string path_mvaWeightFileConvBremEndcapsHighPt_;
      
};
#endif
