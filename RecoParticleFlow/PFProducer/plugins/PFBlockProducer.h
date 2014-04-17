#ifndef RecoParticleFlow_PFProducer_PFBlockProducer_h_
#define RecoParticleFlow_PFProducer_PFBlockProducer_h_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h" 
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


/**\class PFBlockProducer 
\brief Producer for particle flow blocks

This producer makes use of PFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle 
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;



class PFBlockProducer : public edm::stream::EDProducer<> {
 public:

  explicit PFBlockProducer(const edm::ParameterSet&);

  ~PFBlockProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  

  edm::EDGetTokenT<reco::PFRecTrackCollection>  inputTagRecTracks_;
  edm::EDGetTokenT<reco::GsfPFRecTrackCollection>  inputTagGsfRecTracks_;
  edm::EDGetTokenT<reco::GsfPFRecTrackCollection>  inputTagConvBremGsfRecTracks_;
  edm::EDGetTokenT<reco::MuonCollection>  inputTagRecMuons_;
  edm::EDGetTokenT<reco::PFDisplacedTrackerVertexCollection> inputTagPFNuclear_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputTagPFClustersECAL_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputTagPFClustersHCAL_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputTagPFClustersHO_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputTagPFClustersHFEM_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputTagPFClustersHFHAD_;
  edm::EDGetTokenT<reco::PFClusterCollection> inputTagPFClustersPS_;
  edm::EDGetTokenT<reco::PFConversionCollection> inputTagPFConversions_;
  edm::EDGetTokenT<reco::PFV0Collection>   inputTagPFV0_;
  edm::EDGetTokenT<reco::PhotonCollection>  inputTagEGPhotons_;
  edm::EDGetTokenT<reco::SuperClusterCollection>  inputTagSCBarrel_;
  edm::EDGetTokenT<reco::SuperClusterCollection>  inputTagSCEndcap_;
  edm::EDGetTokenT<edm::ValueMap<reco::CaloClusterPtr> > inputTagPFClusterAssociationEBEE_;
  
  // Link track and HCAL clusters to HO clusters ?
  bool useHO_;

  /// verbose ?
  bool   verbose_;

  /// use NuclearInteractions ?
  bool   useNuclear_;

  /// use EG photons ? 
  bool useEGPhotons_;
  
  /// use SuperClusters ? 
  bool useSuperClusters_;  
  
  //match superclusters by ref
  bool superClusterMatchByRef_;
  
  /// switch on/off Conversions
  bool  useConversions_;  
  
  /// switch on/off Conversions Brem Recovery
  bool   useConvBremGsfTracks_;

  /// switch on/off V0
  bool useV0_;

  /// Particle Flow at HLT ?
  bool usePFatHLT_;

  // Glowinski & Gouzevitch
  // Use the optimized KDTree Track/Ecal linker?
  bool useKDTreeTrackEcalLinker_;
  // !Glowinski & Gouzevitch

  /// Particle flow block algorithm 
  PFBlockAlgo            pfBlockAlgo_;

};

DEFINE_FWK_MODULE(PFBlockProducer);

#endif
