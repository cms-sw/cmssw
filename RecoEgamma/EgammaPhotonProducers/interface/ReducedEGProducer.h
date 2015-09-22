#ifndef RecoEgamma_EgammaPhotonProducers_ReducedEGProducer_h
#define RecoEgamma_EgammaPhotonProducers_ReducedEGProducer_h
/** \class ReducedEGProducer
 **  
 **  Select subset of electrons and photons from input collections and
 **  produced consistently relinked output collections including
 **  associated SuperClusters, CaloClusters and ecal RecHits
 **
 **  \author J.Bendavid (CERN)
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/PhotonIdentification/interface/PFPhotonIsolationCalculator.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h" 
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"

#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"


// ReducedEGProducer inherits from EDProducer, so it can be a module:
class ReducedEGProducer : public edm::stream::EDProducer<> {

 public:

  ReducedEGProducer (const edm::ParameterSet& ps);
  ~ReducedEGProducer();

  virtual void produce(edm::Event& evt, const edm::EventSetup& es) override final;

 private: 
  
 //tokens for input collections
 const edm::EDGetTokenT<reco::PhotonCollection> photonT_;
 const edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectronT_; 
 const edm::EDGetTokenT<reco::ConversionCollection> conversionT_;
 const edm::EDGetTokenT<reco::ConversionCollection> singleConversionT_;
 
 const edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
 const edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
 const edm::EDGetTokenT<EcalRecHitCollection> preshowerEcalHits_;
 
 const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > photonPfCandMapT_;
 const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > gsfElectronPfCandMapT_;
 
 std::vector<edm::EDGetTokenT<edm::ValueMap<bool> > > photonIdTs_;
 std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > gsfElectronIdTs_;

 std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > photonPFClusterIsoTs_;
 std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > gsfElectronPFClusterIsoTs_;

 //names for output collections
 const std::string outPhotons_;
 const std::string outPhotonCores_;
 const std::string outGsfElectrons_;
 const std::string outGsfElectronCores_;
 const std::string outConversions_;
 const std::string outSingleConversions_;
 const std::string outSuperClusters_;
 const std::string outEBEEClusters_;
 const std::string outESClusters_;
 const std::string outEBRecHits_;
 const std::string outEERecHits_;
 const std::string outESRecHits_;
 const std::string outPhotonPfCandMap_;
 const std::string outGsfElectronPfCandMap_;
 const std::vector<std::string> outPhotonIds_;
 const std::vector<std::string> outGsfElectronIds_;
 const std::vector<std::string> outPhotonPFClusterIsos_;
 const std::vector<std::string> outGsfElectronPFClusterIsos_;
 
 const StringCutObjectSelector<reco::Photon> keepPhotonSel_;
 const StringCutObjectSelector<reco::Photon> slimRelinkPhotonSel_; 
 const StringCutObjectSelector<reco::Photon> relinkPhotonSel_;
 const StringCutObjectSelector<reco::GsfElectron> keepGsfElectronSel_;
 const StringCutObjectSelector<reco::GsfElectron> slimRelinkGsfElectronSel_;
 const StringCutObjectSelector<reco::GsfElectron> relinkGsfElectronSel_; 
};
#endif


