#ifndef RecoEgamma_EgammaPhotonProducers_GEDPhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_GEDPhotonProducer_h
/** \class GEDPhotonProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"

#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

// GEDPhotonProducer inherits from EDProducer, so it can be a module:
class GEDPhotonProducer : public edm::stream::EDProducer<> {
public:
  GEDPhotonProducer(const edm::ParameterSet& ps);
  ~GEDPhotonProducer() override;

  void beginRun(edm::Run const& r, edm::EventSetup const& es) final;
  void endRun(edm::Run const&, edm::EventSetup const&) final;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  class RecoStepInfo {
  public:
    enum FlagBits { kOOT = 0x1, kFinal = 0x2 };
    explicit RecoStepInfo(const std::string& recoStep);

    bool isOOT() const { return flags_ & kOOT; }
    bool isFinal() const { return flags_ & kFinal; }

  private:
    unsigned int flags_;
  };

  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::PhotonCoreCollection>& photonCoreHandle,
                            const CaloTopology* topology,
                            const EcalRecHitCollection* ecalBarrelHits,
                            const EcalRecHitCollection* ecalEndcapHits,
                            const EcalRecHitCollection* preshowerHits,
                            CaloTowerCollection const& hcalTowers,
                            const reco::VertexCollection& pvVertices,
                            reco::PhotonCollection& outputCollection,
                            int& iSC);

  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::PhotonCollection>& photonHandle,
                            const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
                            const edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle,
                            edm::ValueMap<reco::PhotonRef> pfEGCandToPhotonMap,
                            edm::Handle<reco::VertexCollection>& pvVertices,
                            reco::PhotonCollection& outputCollection,
                            int& iSC,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadrons,
                            const edm::Handle<edm::ValueMap<float>>& neutralHadrons,
                            const edm::Handle<edm::ValueMap<float>>& photons,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadronsWorstVtx,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadronsWorstVtxGeomVeto,
                            const edm::Handle<edm::ValueMap<float>>& chargedHadronsPFPV,
                            const edm::Handle<edm::ValueMap<float>>& pfEcalClusters,
                            const edm::Handle<edm::ValueMap<float>>& pfHcalClusters);

  // std::string PhotonCoreCollection_;
  std::string photonCollection_;
  edm::InputTag photonProducer_;

  edm::EDGetTokenT<reco::PhotonCoreCollection> photonCoreProducerT_;
  edm::EDGetTokenT<reco::PhotonCollection> photonProducerT_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> preshowerHits_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfEgammaCandidates_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
  edm::EDGetTokenT<CaloTowerCollection> hcalTowers_;
  edm::EDGetTokenT<reco::VertexCollection> vertexProducer_;
  //for isolation with map-based veto
  edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> particleBasedIsolationToken;
  //photon isolation sums
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoNeutralHadronIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoPhotonIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedWorstVtxIsoToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedWorstVtxGeomVetoIsoToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoChargedPFPVIsoToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> phoPFECALClusIsolationToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> phoPFHCALClusIsolationToken_;

  std::string conversionProducer_;
  std::string conversionCollection_;
  std::string valueMapPFCandPhoton_;

  PhotonIsolationCalculator* thePhotonIsolationCalculator_;

  //AA
  //Flags and severities to be excluded from calculations

  std::vector<int> flagsexclEB_;
  std::vector<int> flagsexclEE_;
  std::vector<int> severitiesexclEB_;
  std::vector<int> severitiesexclEE_;

  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  double highEt_;
  double minR9Barrel_;
  double minR9Endcap_;
  bool runMIPTagger_;

  bool validConversions_;
  RecoStepInfo recoStep_;

  bool usePrimaryVertex_;
  edm::ParameterSet conf_;
  PositionCalc posCalculator_;

  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::ESHandle<CaloTopology> theCaloTopo_;

  bool validPixelSeeds_;

  //MIP
  PhotonMIPHaloTagger* thePhotonMIPHaloTagger_;

  std::vector<double> preselCutValuesBarrel_;
  std::vector<double> preselCutValuesEndcap_;

  EcalClusterFunctionBaseClass* energyCorrectionF;
  PhotonEnergyCorrector* thePhotonEnergyCorrector_;
  std::string candidateP4type_;

  bool checkHcalStatus_;
};
#endif
