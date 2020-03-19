#ifndef RecoEgamma_EgammaPhotonProducers_ReducedEGProducer_h
#define RecoEgamma_EgammaPhotonProducers_ReducedEGProducer_h
/** \class ReducedEGProducer
 **  
 **  Select subset of electrons and photons from input collections and
 **  produced consistently relinked output collections including
 **  associated SuperClusters, CaloClusters and ecal RecHits
 **
 **  \author J.Bendavid (CERN)
 **  \edited: K. McDermott(Cornell) : refactored code + out of time photons
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
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EGHcalRecHitSelector.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

// ReducedEGProducer inherits from EDProducer, so it can be a module:
class ReducedEGProducer : public edm::stream::EDProducer<> {
public:
  ReducedEGProducer(const edm::ParameterSet& ps);
  ~ReducedEGProducer() override;

  void beginRun(edm::Run const&, const edm::EventSetup&) final;
  void produce(edm::Event& evt, const edm::EventSetup& es) final;

private:
  template <typename T, typename U>
  void linkCore(const T& core, U& cores, std::map<T, unsigned int>& coreMap);

  void linkSuperCluster(const reco::SuperClusterRef& superCluster,
                        std::map<reco::SuperClusterRef, unsigned int>& superClusterMap,
                        reco::SuperClusterCollection& superClusters,
                        const bool relink,
                        std::unordered_set<unsigned int>& superClusterFullRelinkMap);

  void linkConversions(const reco::ConversionRefVector& convrefs,
                       reco::ConversionCollection& conversions,
                       std::map<reco::ConversionRef, unsigned int>& conversionMap);

  void linkConversionsByTrackRef(const edm::Handle<reco::ConversionCollection>& conversionHandle,
                                 const reco::GsfElectron& gsfElectron,
                                 reco::ConversionCollection& conversions,
                                 std::map<reco::ConversionRef, unsigned int>& conversionMap);

  void linkConversionsByTrackRef(const edm::Handle<reco::ConversionCollection>& conversionHandle,
                                 const reco::SuperCluster& superCluster,
                                 reco::ConversionCollection& conversions,
                                 std::map<reco::ConversionRef, unsigned int>& conversionMap);

  void linkConversion(const reco::ConversionRef& convref,
                      reco::ConversionCollection& conversions,
                      std::map<reco::ConversionRef, unsigned int>& conversionMap);

  void linkCaloCluster(const reco::CaloClusterPtr& caloCluster,
                       reco::CaloClusterCollection& caloClusters,
                       std::map<reco::CaloClusterPtr, unsigned int>& caloClusterMap);

  void linkCaloClusters(const reco::SuperCluster& superCluster,
                        reco::CaloClusterCollection& ebeeClusters,
                        std::map<reco::CaloClusterPtr, unsigned int>& ebeeClusterMap,
                        std::unordered_set<DetId>& rechitMap,
                        const edm::Handle<EcalRecHitCollection>& barrelHitHandle,
                        const edm::Handle<EcalRecHitCollection>& endcapHitHandle,
                        const CaloTopology* caloTopology,
                        reco::CaloClusterCollection& esClusters,
                        std::map<reco::CaloClusterPtr, unsigned int>& esClusterMap);

  void linkHcalHits(const reco::SuperCluster& superClus,
                    const HBHERecHitCollection& recHits,
                    std::unordered_set<DetId>& hcalDetIds);

  void relinkCaloClusters(reco::SuperCluster& superCluster,
                          const std::map<reco::CaloClusterPtr, unsigned int>& ebeeClusterMap,
                          const std::map<reco::CaloClusterPtr, unsigned int>& esClusterMap,
                          const edm::OrphanHandle<reco::CaloClusterCollection>& outEBEEClusterHandle,
                          const edm::OrphanHandle<reco::CaloClusterCollection>& outESClusterHandle);

  template <typename T>
  void relinkSuperCluster(T& core,
                          const std::map<reco::SuperClusterRef, unsigned int>& superClusterMap,
                          const edm::OrphanHandle<reco::SuperClusterCollection>& outSuperClusterHandle);

  void relinkGsfTrack(reco::GsfElectronCore& electroncore,
                      const std::map<reco::GsfTrackRef, unsigned int>& gsfTrackMap,
                      const edm::OrphanHandle<reco::GsfTrackCollection>& outGsfTrackHandle);

  void relinkConversions(reco::PhotonCore& photonCore,
                         const reco::ConversionRefVector& convrefs,
                         const std::map<reco::ConversionRef, unsigned int>& conversionMap,
                         const edm::OrphanHandle<reco::ConversionCollection>& outConversionHandle);

  void relinkPhotonCore(reco::Photon& photon,
                        const std::map<reco::PhotonCoreRef, unsigned int>& photonCoreMap,
                        const edm::OrphanHandle<reco::PhotonCoreCollection>& outPhotonCoreHandle);

  void relinkGsfElectronCore(reco::GsfElectron& gsfElectron,
                             const std::map<reco::GsfElectronCoreRef, unsigned int>& gsfElectronCoreMap,
                             const edm::OrphanHandle<reco::GsfElectronCoreCollection>& outGsfElectronCoreHandle);

  static void calibratePhoton(reco::Photon& photon,
                              const reco::PhotonRef& oldPhoRef,
                              const edm::ValueMap<float>& energyMap,
                              const edm::ValueMap<float>& energyErrMap);

  static void calibrateElectron(reco::GsfElectron& gsfElectron,
                                const reco::GsfElectronRef& oldEleRef,
                                const edm::ValueMap<float>& energyMap,
                                const edm::ValueMap<float>& energyErrMap,
                                const edm::ValueMap<float>& ecalEnergyMap,
                                const edm::ValueMap<float>& ecalEnergyErrMap);

  template <typename T>
  void setToken(edm::EDGetTokenT<T>& token, const edm::ParameterSet& config, const std::string& name) {
    token = consumes<T>(config.getParameter<edm::InputTag>(name));
  }

  //tokens for input collections
  const edm::EDGetTokenT<reco::PhotonCollection> photonT_;
  edm::EDGetTokenT<reco::PhotonCollection> ootPhotonT_;
  const edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectronT_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> gsfTrackT_;
  const edm::EDGetTokenT<reco::ConversionCollection> conversionT_;
  const edm::EDGetTokenT<reco::ConversionCollection> singleConversionT_;

  const edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  const bool doPreshowerEcalHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> preshowerEcalHits_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheHits_;

  const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > photonPfCandMapT_;
  const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > gsfElectronPfCandMapT_;

  std::vector<edm::EDGetTokenT<edm::ValueMap<bool> > > photonIdTs_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > gsfElectronIdTs_;

  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > photonFloatValueMapTs_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > ootPhotonFloatValueMapTs_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > gsfElectronFloatValueMapTs_;

  const bool applyPhotonCalibOnData_;
  const bool applyPhotonCalibOnMC_;
  const bool applyGsfElectronCalibOnData_;
  const bool applyGsfElectronCalibOnMC_;
  edm::EDGetTokenT<edm::ValueMap<float> > photonCalibEnergyT_;
  edm::EDGetTokenT<edm::ValueMap<float> > photonCalibEnergyErrT_;
  edm::EDGetTokenT<edm::ValueMap<float> > gsfElectronCalibEnergyT_;
  edm::EDGetTokenT<edm::ValueMap<float> > gsfElectronCalibEnergyErrT_;
  edm::EDGetTokenT<edm::ValueMap<float> > gsfElectronCalibEcalEnergyT_;
  edm::EDGetTokenT<edm::ValueMap<float> > gsfElectronCalibEcalEnergyErrT_;

  //names for output collections
  const std::string outPhotons_;
  const std::string outPhotonCores_;
  const std::string outOOTPhotons_;
  const std::string outOOTPhotonCores_;
  const std::string outGsfElectrons_;
  const std::string outGsfElectronCores_;
  const std::string outGsfTracks_;
  const std::string outConversions_;
  const std::string outSingleConversions_;
  const std::string outSuperClusters_;
  const std::string outEBEEClusters_;
  const std::string outESClusters_;
  const std::string outOOTSuperClusters_;
  const std::string outOOTEBEEClusters_;
  const std::string outOOTESClusters_;
  const std::string outEBRecHits_;
  const std::string outEERecHits_;
  const std::string outESRecHits_;
  const std::string outHBHERecHits_;
  const std::string outPhotonPfCandMap_;
  const std::string outGsfElectronPfCandMap_;
  const std::vector<std::string> outPhotonIds_;
  const std::vector<std::string> outGsfElectronIds_;
  const std::vector<std::string> outPhotonFloatValueMaps_;
  const std::vector<std::string> outOOTPhotonFloatValueMaps_;
  const std::vector<std::string> outGsfElectronFloatValueMaps_;

  const StringCutObjectSelector<reco::Photon> keepPhotonSel_;
  const StringCutObjectSelector<reco::Photon> slimRelinkPhotonSel_;
  const StringCutObjectSelector<reco::Photon> relinkPhotonSel_;
  const StringCutObjectSelector<reco::Photon> keepOOTPhotonSel_;
  const StringCutObjectSelector<reco::Photon> slimRelinkOOTPhotonSel_;
  const StringCutObjectSelector<reco::Photon> relinkOOTPhotonSel_;
  const StringCutObjectSelector<reco::GsfElectron> keepGsfElectronSel_;
  const StringCutObjectSelector<reco::GsfElectron> slimRelinkGsfElectronSel_;
  const StringCutObjectSelector<reco::GsfElectron> relinkGsfElectronSel_;

  EGHcalRecHitSelector hcalHitSel_;
};
#endif
