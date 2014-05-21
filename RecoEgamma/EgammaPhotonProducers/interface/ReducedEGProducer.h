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
#include "RecoEgamma/EgammaIsolationAlgos/interface/PfBlockBasedIsolation.h"
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
class ReducedEGProducer : public edm::EDProducer {

 public:

  ReducedEGProducer (const edm::ParameterSet& ps);
  ~ReducedEGProducer();

  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private: 
  
 //tokens for input collections
 edm::EDGetTokenT<reco::PhotonCollection> photonT_;
 edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectronT_; 
 edm::EDGetTokenT<reco::ConversionCollection> conversionT_;
 edm::EDGetTokenT<reco::ConversionCollection> singleConversionT_;
 
 edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
 edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
 edm::EDGetTokenT<EcalRecHitCollection> preshowerEcalHits_;
 
 edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > photonPfCandMapT_;
 edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > gsfElectronPfCandMapT_;
 
 std::vector<edm::EDGetTokenT<edm::ValueMap<bool> > > photonIdTs_;
 std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > gsfElectronIdTs_;
 
 //names for output collections
 std::string outPhotons_;
 std::string outPhotonCores_;
 std::string outGsfElectrons_;
 std::string outGsfElectronCores_;
 std::string outConversions_;
 std::string outSingleConversions_;
 std::string outSuperClusters_;
 std::string outEBEEClusters_;
 std::string outESClusters_;
 std::string outEBRecHits_;
 std::string outEERecHits_;
 std::string outESRecHits_;
 std::string outPhotonPfCandMap_;
 std::string outGsfElectronPfCandMap_;
 std::vector<std::string> outPhotonIds_;
 std::vector<std::string> outGsfElectronIds_;
 
 StringCutObjectSelector<reco::Photon> keepPhotonSel_;
 StringCutObjectSelector<reco::Photon> slimRelinkPhotonSel_; 
 StringCutObjectSelector<reco::Photon> relinkPhotonSel_;
 StringCutObjectSelector<reco::GsfElectron> keepGsfElectronSel_;
 StringCutObjectSelector<reco::GsfElectron> slimRelinkGsfElectronSel_;
 StringCutObjectSelector<reco::GsfElectron> relinkGsfElectronSel_; 
 
 

};
#endif


