#ifndef RecoParticleFlow_PFProducer_PFPhotonTranslator_H
#define RecoParticleFlow_PFProducer_PFPhotonTranslator_H
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"

#include <iostream>
#include <string>
#include <map>
#include <vector>

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"

#include <Math/VectorUtil.h>
#include "TLorentzVector.h"
#include "TMath.h"

class PFPhotonTranslator : public edm::EDProducer
{
 public:
  explicit PFPhotonTranslator(const edm::ParameterSet&);
  ~PFPhotonTranslator();
  
  virtual void produce(edm::Event &, const edm::EventSetup&) override;

  typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;

 private:
  // to retrieve the collection from the event
  bool fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
				const edm::InputTag& tag, 
				const edm::Event& iEvent) const;


  // makes a basic cluster from PFBlockElement and add it to the collection ; the corrected energy is taken
  // from the PFCandidate
  void createBasicCluster(const reco::PFBlockElement & ,  reco::BasicClusterCollection & basicClusters,
			  std::vector<const reco::PFCluster *> &,
			  const reco::PFCandidate & coCandidate) const;
  // makes a preshower cluster from of PFBlockElement and add it to the collection
  void createPreshowerCluster(const reco::PFBlockElement & PFBE, 
			      reco::PreshowerClusterCollection& preshowerClusters,
			      unsigned plane) const;
  
  // create the basic cluster Ptr
  void createBasicClusterPtrs(const edm::OrphanHandle<reco::BasicClusterCollection> & basicClustersHandle );

  // create the preshower cluster Refs
  void createPreshowerClusterPtrs(const edm::OrphanHandle<reco::PreshowerClusterCollection> & preshowerClustersHandle );

  // make a super cluster from its ingredients and add it to the collection
  void createSuperClusters(const reco::PFCandidateCollection &,
			  reco::SuperClusterCollection &superClusters) const;
  
  void createOneLegConversions(const edm::OrphanHandle<reco::SuperClusterCollection> & superClustersHandle, reco::ConversionCollection &oneLegConversions);

  //create photon cores
  void createPhotonCores(const edm::OrphanHandle<reco::SuperClusterCollection> & superClustersHandle, const edm::OrphanHandle<reco::ConversionCollection> & oneLegConversionHandle, reco::PhotonCoreCollection &photonCores) ;

  //create photons
  //void createPhotons(reco::VertexCollection &vertexCollection, const edm::OrphanHandle<reco::PhotonCoreCollection> & superClustersHandle, const CaloTopology* topology, const EcalRecHitCollection * barrelRecHits, const EcalRecHitCollection * endcapRecHits, const edm::Handle<CaloTowerCollection> & hcalTowersHandle, const IsolationValueMaps& isolationValues, reco::PhotonCollection &photons) ;
  void createPhotons(reco::VertexCollection &vertexCollection, edm::Handle<reco::PhotonCollection> &egPhotons, const edm::OrphanHandle<reco::PhotonCoreCollection> & photonCoresHandle, const IsolationValueMaps& isolationValues, reco::PhotonCollection &photons) ;

  const reco::PFCandidate & correspondingDaughterCandidate(const reco::PFCandidate & cand, const reco::PFBlockElement & pfbe) const;

  edm::InputTag inputTagPFCandidates_;
  std::vector<edm::InputTag> inputTagIsoVals_;
  std::string PFBasicClusterCollection_;
  std::string PFPreshowerClusterCollection_;
  std::string PFSuperClusterCollection_;
  std::string PFPhotonCoreCollection_;
  std::string PFPhotonCollection_;
  std::string PFConversionCollection_;
  std::string EGPhotonCollection_;
  std::string  vertexProducer_;
  edm::InputTag barrelEcalHits_;
  edm::InputTag endcapEcalHits_;
  edm::InputTag hcalTowers_;
  double hOverEConeSize_;

  // the collection of basic clusters associated to a photon
  std::vector<reco::BasicClusterCollection> basicClusters_;
  // the correcsponding PFCluster ref
  std::vector<std::vector<const reco::PFCluster *> > pfClusters_;
  // the collection of preshower clusters associated to a photon
  std::vector<reco::PreshowerClusterCollection> preshowerClusters_;
  // the super cluster collection (actually only one) associated to a photon
  std::vector<reco::SuperClusterCollection> superClusters_;
  // the references to the basic clusters associated to a photon
  std::vector<reco::CaloClusterPtrVector> basicClusterPtr_;
  // the references to the basic clusters associated to a photon
  std::vector<reco::CaloClusterPtrVector> preshowerClusterPtr_;
   // keep track of the index of the PF Candidate
  std::vector<int> photPFCandidateIndex_;
  // the list of candidatePtr
  std::vector<reco::CandidatePtr> CandidatePtr_;
  // the e/g SC associated
  std::vector<reco::SuperClusterRef> egSCRef_;
  // the e/g photon associated
  std::vector<reco::PhotonRef> egPhotonRef_;
  // the PF MVA and regression 
  std::vector<float> pfPhotonMva_;
  std::vector<float> energyRegression_;
  std::vector<float> energyRegressionError_;

  //Vector of vector of Conversions Refs
  std::vector<reco::ConversionRefVector > pfConv_;
  std::vector< std::vector<reco::TrackRef> > pfSingleLegConv_;
  std::vector< std::vector<float> > pfSingleLegConvMva_;
  //std::vector<reco::TrackRef> * pfSingleLegConv_;

  std::vector<int> conv1legPFCandidateIndex_;
  std::vector<int> conv2legPFCandidateIndex_;

  edm::ESHandle<CaloTopology> theCaloTopo_;
  edm::ESHandle<CaloGeometry> theCaloGeom_;

  bool emptyIsOk_;

};
#endif
