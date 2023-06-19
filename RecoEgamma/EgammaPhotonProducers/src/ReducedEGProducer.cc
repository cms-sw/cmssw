/** \class ReducedEGProducer
 **  
 **  Select subset of electrons and photons from input collections and
 **  produced consistently relinked output collections including
 **  associated SuperClusters, CaloClusters and ecal RecHits
 **
 **  \author J.Bendavid (CERN)
 **  \edited: K. McDermott(Cornell) : refactored code + out of time photons
 ***/

#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/HIPhotonIsolation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SlimmedSuperCluster.h"
#include "DataFormats/EgammaReco/interface/SlimmedSuperClusterFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EGHcalRecHitSelector.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHLTTrackIsolation.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include <memory>
#include <unordered_set>
#include <vector>

class ReducedEGProducer : public edm::stream::EDProducer<> {
public:
  ReducedEGProducer(const edm::ParameterSet& ps);

  void beginRun(edm::Run const&, const edm::EventSetup&) final;
  void produce(edm::Event& evt, const edm::EventSetup& es) final;

private:
  template <class T>
  std::vector<edm::EDPutTokenT<T>> vproduces(std::vector<std::string> const& labels) {
    std::vector<edm::EDPutTokenT<T>> putTokens{};
    putTokens.reserve(labels.size());
    for (const auto& label : labels) {
      putTokens.push_back(produces<T>(label));
    }
    return putTokens;
  }

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
                        CaloTopology const& caloTopology,
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

  std::unique_ptr<reco::SlimmedSuperClusterCollection> produceSlimmedSuperClusters(const edm::Event& event) const;

  template <typename T>
  void setToken(edm::EDGetTokenT<T>& token, const edm::ParameterSet& config, const std::string& name) {
    token = consumes<T>(config.getParameter<edm::InputTag>(name));
  }

  //tokens for input collections
  const edm::EDGetTokenT<reco::PhotonCollection> photonT_;
  edm::EDGetTokenT<reco::PhotonCollection> ootPhotonT_;
  const edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectronT_;
  const edm::EDGetTokenT<reco::ConversionCollection> conversionT_;
  const edm::EDGetTokenT<reco::ConversionCollection> singleConversionT_;

  const edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  const bool doPreshowerEcalHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> preshowerEcalHits_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheHits_;

  const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> photonPfCandMapT_;
  const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> gsfElectronPfCandMapT_;

  std::vector<edm::EDGetTokenT<edm::ValueMap<bool>>> photonIdTs_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> gsfElectronIdTs_;

  std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> photonFloatValueMapTs_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> ootPhotonFloatValueMapTs_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float>>> gsfElectronFloatValueMapTs_;

  const edm::EDGetTokenT<reco::HIPhotonIsolationMap> recoHIPhotonIsolationMapInputToken_;
  edm::EDPutTokenT<reco::HIPhotonIsolationMap> recoHIPhotonIsolationMapOutputName_;

  const bool applyPhotonCalibOnData_;
  const bool applyPhotonCalibOnMC_;
  const bool applyGsfElectronCalibOnData_;
  const bool applyGsfElectronCalibOnMC_;
  edm::EDGetTokenT<edm::ValueMap<float>> photonCalibEnergyT_;
  edm::EDGetTokenT<edm::ValueMap<float>> photonCalibEnergyErrT_;
  edm::EDGetTokenT<edm::ValueMap<float>> gsfElectronCalibEnergyT_;
  edm::EDGetTokenT<edm::ValueMap<float>> gsfElectronCalibEnergyErrT_;
  edm::EDGetTokenT<edm::ValueMap<float>> gsfElectronCalibEcalEnergyT_;
  edm::EDGetTokenT<edm::ValueMap<float>> gsfElectronCalibEcalEnergyErrT_;

  std::vector<edm::EDGetTokenT<reco::SuperClusterCollection>> superClustersToSaveTs_;
  const edm::EDGetTokenT<reco::TrackCollection> trksForSCIso_;

  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopology_;
  //names for output collections
  const edm::EDPutTokenT<reco::PhotonCollection> outPhotons_;
  const edm::EDPutTokenT<reco::PhotonCoreCollection> outPhotonCores_;
  edm::EDPutTokenT<reco::PhotonCollection> outOOTPhotons_;
  edm::EDPutTokenT<reco::PhotonCoreCollection> outOOTPhotonCores_;
  const edm::EDPutTokenT<reco::GsfElectronCollection> outGsfElectrons_;
  const edm::EDPutTokenT<reco::GsfElectronCoreCollection> outGsfElectronCores_;
  const edm::EDPutTokenT<reco::GsfTrackCollection> outGsfTracks_;
  const edm::EDPutTokenT<reco::ConversionCollection> outConversions_;
  const edm::EDPutTokenT<reco::ConversionCollection> outSingleConversions_;
  const edm::EDPutTokenT<reco::SuperClusterCollection> outSuperClusters_;
  const edm::EDPutTokenT<reco::CaloClusterCollection> outEBEEClusters_;
  const edm::EDPutTokenT<reco::CaloClusterCollection> outESClusters_;
  edm::EDPutTokenT<reco::SuperClusterCollection> outOOTSuperClusters_;
  edm::EDPutTokenT<reco::CaloClusterCollection> outOOTEBEEClusters_;
  edm::EDPutTokenT<reco::CaloClusterCollection> outOOTESClusters_;
  const edm::EDPutTokenT<EcalRecHitCollection> outEBRecHits_;
  const edm::EDPutTokenT<EcalRecHitCollection> outEERecHits_;
  edm::EDPutTokenT<EcalRecHitCollection> outESRecHits_;
  const edm::EDPutTokenT<HBHERecHitCollection> outHBHERecHits_;
  const edm::EDPutTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> outPhotonPfCandMap_;
  const edm::EDPutTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> outGsfElectronPfCandMap_;
  const std::vector<edm::EDPutTokenT<edm::ValueMap<bool>>> outPhotonIds_;
  const std::vector<edm::EDPutTokenT<edm::ValueMap<float>>> outGsfElectronIds_;
  const std::vector<edm::EDPutTokenT<edm::ValueMap<float>>> outPhotonFloatValueMaps_;
  std::vector<edm::EDPutTokenT<edm::ValueMap<float>>> outOOTPhotonFloatValueMaps_;
  const std::vector<edm::EDPutTokenT<edm::ValueMap<float>>> outGsfElectronFloatValueMaps_;
  const edm::EDPutTokenT<reco::SlimmedSuperClusterCollection> outSlimmedSuperClusters_;

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
  EgammaHLTTrackIsolation trkIsoCalc_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ReducedEGProducer);

namespace {

  template <class T>
  auto getHandles(edm::Event const& event, std::vector<edm::EDGetTokenT<T>> const& tokens) {
    std::vector<edm::Handle<T>> handles(tokens.size());
    int index = 0;
    for (const auto& token : tokens) {
      event.getByToken(token, handles[index++]);
    }
    return handles;
  }

  template <class Handle, class T>
  auto emplaceValueMap(Handle const& handle,
                       std::vector<T> const& values,
                       edm::Event& ev,
                       edm::EDPutTokenT<edm::ValueMap<T>> const& putToken) {
    using MapType = edm::ValueMap<T>;
    MapType oMap{};
    {
      typename MapType::Filler filler(oMap);
      filler.insert(handle, values.begin(), values.end());
      filler.fill();
    }
    ev.emplace(putToken, std::move(oMap));
  };
}  // namespace

ReducedEGProducer::ReducedEGProducer(const edm::ParameterSet& config)
    : photonT_(consumes(config.getParameter<edm::InputTag>("photons"))),
      gsfElectronT_(consumes(config.getParameter<edm::InputTag>("gsfElectrons"))),
      conversionT_(consumes(config.getParameter<edm::InputTag>("conversions"))),
      singleConversionT_(consumes(config.getParameter<edm::InputTag>("singleConversions"))),
      barrelEcalHits_(consumes(config.getParameter<edm::InputTag>("barrelEcalHits"))),
      endcapEcalHits_(consumes(config.getParameter<edm::InputTag>("endcapEcalHits"))),
      doPreshowerEcalHits_(!config.getParameter<edm::InputTag>("preshowerEcalHits").label().empty()),
      preshowerEcalHits_(doPreshowerEcalHits_
                             ? consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("preshowerEcalHits"))
                             : edm::EDGetTokenT<EcalRecHitCollection>()),
      hbheHits_(consumes<HBHERecHitCollection>(config.getParameter<edm::InputTag>("hbheHits"))),
      photonPfCandMapT_(consumes(config.getParameter<edm::InputTag>("photonsPFValMap"))),
      gsfElectronPfCandMapT_(consumes(config.getParameter<edm::InputTag>("gsfElectronsPFValMap"))),
      recoHIPhotonIsolationMapInputToken_{
          !config.getParameter<edm::InputTag>("hiPhotonIsolationMapInput").label().empty()
              ? consumes<reco::HIPhotonIsolationMap>(config.getParameter<edm::InputTag>("hiPhotonIsolationMapInput"))
              : edm::EDGetTokenT<reco::HIPhotonIsolationMap>{}},
      //calibration flags
      applyPhotonCalibOnData_(config.getParameter<bool>("applyPhotonCalibOnData")),
      applyPhotonCalibOnMC_(config.getParameter<bool>("applyPhotonCalibOnMC")),
      applyGsfElectronCalibOnData_(config.getParameter<bool>("applyGsfElectronCalibOnData")),
      applyGsfElectronCalibOnMC_(config.getParameter<bool>("applyGsfElectronCalibOnMC")),
      trksForSCIso_(consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("trksForSCIso"))),
      //output collections
      outPhotons_{produces<reco::PhotonCollection>("reducedGedPhotons")},
      outPhotonCores_{produces<reco::PhotonCoreCollection>("reducedGedPhotonCores")},
      outGsfElectrons_{produces<reco::GsfElectronCollection>("reducedGedGsfElectrons")},
      outGsfElectronCores_{produces<reco::GsfElectronCoreCollection>("reducedGedGsfElectronCores")},
      outGsfTracks_{produces<reco::GsfTrackCollection>("reducedGsfTracks")},
      outConversions_{produces<reco::ConversionCollection>("reducedConversions")},
      outSingleConversions_{produces<reco::ConversionCollection>("reducedSingleLegConversions")},
      outSuperClusters_{produces<reco::SuperClusterCollection>("reducedSuperClusters")},
      outEBEEClusters_{produces<reco::CaloClusterCollection>("reducedEBEEClusters")},
      outESClusters_{produces<reco::CaloClusterCollection>("reducedESClusters")},
      outEBRecHits_{produces<EcalRecHitCollection>("reducedEBRecHits")},
      outEERecHits_{produces<EcalRecHitCollection>("reducedEERecHits")},
      outHBHERecHits_{produces<HBHERecHitCollection>("reducedHBHEHits")},
      outPhotonPfCandMap_{produces<edm::ValueMap<std::vector<reco::PFCandidateRef>>>("reducedPhotonPfCandMap")},
      outGsfElectronPfCandMap_{
          produces<edm::ValueMap<std::vector<reco::PFCandidateRef>>>("reducedGsfElectronPfCandMap")},
      outPhotonIds_{vproduces<edm::ValueMap<bool>>(config.getParameter<std::vector<std::string>>("photonIDOutput"))},
      outGsfElectronIds_{
          vproduces<edm::ValueMap<float>>(config.getParameter<std::vector<std::string>>("gsfElectronIDOutput"))},
      outPhotonFloatValueMaps_{
          vproduces<edm::ValueMap<float>>(config.getParameter<std::vector<std::string>>("photonFloatValueMapOutput"))},
      outGsfElectronFloatValueMaps_{vproduces<edm::ValueMap<float>>(
          config.getParameter<std::vector<std::string>>("gsfElectronFloatValueMapOutput"))},
      outSlimmedSuperClusters_{produces<reco::SlimmedSuperClusterCollection>("slimmedSuperClusters")},
      keepPhotonSel_(config.getParameter<std::string>("keepPhotons")),
      slimRelinkPhotonSel_(config.getParameter<std::string>("slimRelinkPhotons")),
      relinkPhotonSel_(config.getParameter<std::string>("relinkPhotons")),
      keepOOTPhotonSel_(config.getParameter<std::string>("keepOOTPhotons")),
      slimRelinkOOTPhotonSel_(config.getParameter<std::string>("slimRelinkOOTPhotons")),
      relinkOOTPhotonSel_(config.getParameter<std::string>("relinkOOTPhotons")),
      keepGsfElectronSel_(config.getParameter<std::string>("keepGsfElectrons")),
      slimRelinkGsfElectronSel_(config.getParameter<std::string>("slimRelinkGsfElectrons")),
      relinkGsfElectronSel_(config.getParameter<std::string>("relinkGsfElectrons")),
      hcalHitSel_(config.getParameter<edm::ParameterSet>("hcalHitSel"), consumesCollector()),
      trkIsoCalc_(config.getParameter<edm::ParameterSet>("scTrkIsol")) {
  const auto& aTag = config.getParameter<edm::InputTag>("ootPhotons");
  caloTopology_ = esConsumes();
  if (not aTag.label().empty())
    ootPhotonT_ = consumes<reco::PhotonCollection>(aTag);

  for (const edm::InputTag& tag : config.getParameter<std::vector<edm::InputTag>>("photonIDSources")) {
    photonIdTs_.emplace_back(consumes<edm::ValueMap<bool>>(tag));
  }

  for (const edm::InputTag& tag : config.getParameter<std::vector<edm::InputTag>>("gsfElectronIDSources")) {
    gsfElectronIdTs_.emplace_back(consumes<edm::ValueMap<float>>(tag));
  }

  for (const edm::InputTag& tag : config.getParameter<std::vector<edm::InputTag>>("photonFloatValueMapSources")) {
    photonFloatValueMapTs_.emplace_back(consumes<edm::ValueMap<float>>(tag));
  }

  for (const edm::InputTag& tag : config.getParameter<std::vector<edm::InputTag>>("ootPhotonFloatValueMapSources")) {
    ootPhotonFloatValueMapTs_.emplace_back(consumes<edm::ValueMap<float>>(tag));
  }

  for (const edm::InputTag& tag : config.getParameter<std::vector<edm::InputTag>>("gsfElectronFloatValueMapSources")) {
    gsfElectronFloatValueMapTs_.emplace_back(consumes<edm::ValueMap<float>>(tag));
  }

  if (applyPhotonCalibOnData_ || applyPhotonCalibOnMC_) {
    setToken(photonCalibEnergyT_, config, "photonCalibEnergySource");
    setToken(photonCalibEnergyErrT_, config, "photonCalibEnergyErrSource");
  }
  if (applyGsfElectronCalibOnData_ || applyGsfElectronCalibOnMC_) {
    setToken(gsfElectronCalibEnergyT_, config, "gsfElectronCalibEnergySource");
    setToken(gsfElectronCalibEnergyErrT_, config, "gsfElectronCalibEnergyErrSource");
    setToken(gsfElectronCalibEcalEnergyT_, config, "gsfElectronCalibEcalEnergySource");
    setToken(gsfElectronCalibEcalEnergyErrT_, config, "gsfElectronCalibEcalEnergyErrSource");
  }

  for (const auto& tag : config.getParameter<std::vector<edm::InputTag>>("superClustersToSlim")) {
    superClustersToSaveTs_.emplace_back(consumes<reco::SuperClusterCollection>(tag));
  }

  if (!ootPhotonT_.isUninitialized()) {
    outOOTPhotons_ = produces<reco::PhotonCollection>("reducedOOTPhotons");
    outOOTPhotonCores_ = produces<reco::PhotonCoreCollection>("reducedOOTPhotonCores");
    outOOTSuperClusters_ = produces<reco::SuperClusterCollection>("reducedOOTSuperClusters");
    outOOTEBEEClusters_ = produces<reco::CaloClusterCollection>("reducedOOTEBEEClusters");
    outOOTESClusters_ = produces<reco::CaloClusterCollection>("reducedOOTESClusters");
  }
  if (doPreshowerEcalHits_) {
    outESRecHits_ = produces<EcalRecHitCollection>("reducedESRecHits");
  }
  if (!ootPhotonT_.isUninitialized()) {
    outOOTPhotonFloatValueMaps_ =
        vproduces<edm::ValueMap<float>>(config.getParameter<std::vector<std::string>>("ootPhotonFloatValueMapOutput"));
  }
  if (!recoHIPhotonIsolationMapInputToken_.isUninitialized()) {
    recoHIPhotonIsolationMapOutputName_ =
        produces<reco::HIPhotonIsolationMap>(config.getParameter<std::string>("hiPhotonIsolationMapOutput"));
  }
}

void ReducedEGProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup) { hcalHitSel_.setup(iSetup); }

void ReducedEGProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  //get input collections

  auto photonHandle = event.getHandle(photonT_);

  auto ootPhotonHandle =
      !ootPhotonT_.isUninitialized() ? event.getHandle(ootPhotonT_) : edm::Handle<reco::PhotonCollection>{};

  auto gsfElectronHandle = event.getHandle(gsfElectronT_);
  auto conversionHandle = event.getHandle(conversionT_);
  auto singleConversionHandle = event.getHandle(singleConversionT_);
  auto barrelHitHandle = event.getHandle(barrelEcalHits_);
  auto endcapHitHandle = event.getHandle(endcapEcalHits_);

  auto preshowerHitHandle =
      doPreshowerEcalHits_ ? event.getHandle(preshowerEcalHits_) : edm::Handle<EcalRecHitCollection>{};

  auto hbheHitHandle = event.getHandle(hbheHits_);
  auto photonPfCandMapHandle = event.getHandle(photonPfCandMapT_);
  auto gsfElectronPfCandMapHandle = event.getHandle(gsfElectronPfCandMapT_);

  auto photonIdHandles = getHandles(event, photonIdTs_);
  auto gsfElectronIdHandles = getHandles(event, gsfElectronIdTs_);
  auto photonFloatValueMapHandles = getHandles(event, photonFloatValueMapTs_);

  auto ootPhotonFloatValueMapHandles = !ootPhotonT_.isUninitialized()
                                           ? getHandles(event, ootPhotonFloatValueMapTs_)
                                           : std::vector<edm::Handle<edm::ValueMap<float>>>{};

  auto gsfElectronFloatValueMapHandles = getHandles(event, gsfElectronFloatValueMapTs_);

  edm::Handle<edm::ValueMap<float>> gsfElectronCalibEnergyHandle;
  edm::Handle<edm::ValueMap<float>> gsfElectronCalibEnergyErrHandle;
  edm::Handle<edm::ValueMap<float>> gsfElectronCalibEcalEnergyHandle;
  edm::Handle<edm::ValueMap<float>> gsfElectronCalibEcalEnergyErrHandle;
  if (applyGsfElectronCalibOnData_ || applyGsfElectronCalibOnMC_) {
    event.getByToken(gsfElectronCalibEnergyT_, gsfElectronCalibEnergyHandle);
    event.getByToken(gsfElectronCalibEnergyErrT_, gsfElectronCalibEnergyErrHandle);
    event.getByToken(gsfElectronCalibEcalEnergyT_, gsfElectronCalibEcalEnergyHandle);
    event.getByToken(gsfElectronCalibEcalEnergyErrT_, gsfElectronCalibEcalEnergyErrHandle);
  }
  edm::Handle<edm::ValueMap<float>> photonCalibEnergyHandle;
  edm::Handle<edm::ValueMap<float>> photonCalibEnergyErrHandle;
  if (applyPhotonCalibOnData_ || applyPhotonCalibOnMC_) {
    event.getByToken(photonCalibEnergyT_, photonCalibEnergyHandle);
    event.getByToken(photonCalibEnergyErrT_, photonCalibEnergyErrHandle);
  }

  auto const& caloTopology = eventSetup.getData(caloTopology_);

  //initialize output collections
  reco::PhotonCollection photons;
  reco::PhotonCoreCollection photonCores;
  reco::PhotonCollection ootPhotons;
  reco::PhotonCoreCollection ootPhotonCores;
  reco::GsfElectronCollection gsfElectrons;
  reco::GsfElectronCoreCollection gsfElectronCores;
  reco::GsfTrackCollection gsfTracks;
  reco::ConversionCollection conversions;
  reco::ConversionCollection singleConversions;
  reco::SuperClusterCollection superClusters;
  reco::CaloClusterCollection ebeeClusters;
  reco::CaloClusterCollection esClusters;
  reco::SuperClusterCollection ootSuperClusters;
  reco::CaloClusterCollection ootEbeeClusters;
  reco::CaloClusterCollection ootEsClusters;
  EcalRecHitCollection ebRecHits;
  EcalRecHitCollection eeRecHits;
  EcalRecHitCollection esRecHits;
  HBHERecHitCollection hbheRecHits;
  reco::SlimmedSuperClusterCollection slimmedMustacheSuperClusters;
  edm::ValueMap<std::vector<reco::PFCandidateRef>> photonPfCandMap;
  edm::ValueMap<std::vector<reco::PFCandidateRef>> gsfElectronPfCandMap;

  //maps to collection indices of output objects
  std::map<reco::PhotonCoreRef, unsigned int> photonCoreMap;
  std::map<reco::PhotonCoreRef, unsigned int> ootPhotonCoreMap;
  std::map<reco::GsfElectronCoreRef, unsigned int> gsfElectronCoreMap;
  std::map<reco::GsfTrackRef, unsigned int> gsfTrackMap;
  std::map<reco::ConversionRef, unsigned int> conversionMap;
  std::map<reco::ConversionRef, unsigned int> singleConversionMap;
  std::map<reco::SuperClusterRef, unsigned int> superClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> ebeeClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> esClusterMap;
  std::map<reco::SuperClusterRef, unsigned int> ootSuperClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> ootEbeeClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> ootEsClusterMap;
  std::unordered_set<DetId> rechitMap;
  std::unordered_set<DetId> hcalRechitMap;

  std::unordered_set<unsigned int> superClusterFullRelinkMap;
  std::unordered_set<unsigned int> ootSuperClusterFullRelinkMap;

  //vectors for pfcandidate valuemaps
  std::vector<std::vector<reco::PFCandidateRef>> pfCandIsoPairVecPho;
  std::vector<std::vector<reco::PFCandidateRef>> pfCandIsoPairVecEle;

  //vectors for id valuemaps
  std::vector<std::vector<bool>> photonIdVals(photonIdHandles.size());
  std::vector<std::vector<float>> gsfElectronIdVals(gsfElectronIdHandles.size());
  std::vector<std::vector<float>> photonFloatValueMapVals(photonFloatValueMapHandles.size());
  std::vector<std::vector<float>> ootPhotonFloatValueMapVals(ootPhotonFloatValueMapHandles.size());
  std::vector<std::vector<float>> gsfElectronFloatValueMapVals(gsfElectronFloatValueMapHandles.size());

  // HI photon iso value maps
  reco::HIPhotonIsolationMap const* recoHIPhotonIsolationMapInputValueMap =
      !recoHIPhotonIsolationMapInputToken_.isUninitialized() ? &event.get(recoHIPhotonIsolationMapInputToken_)
                                                             : nullptr;
  std::vector<reco::HIPhotonIsolation> recoHIPhotonIsolationMapInputVals;

  //loop over photons and fill maps
  int index = -1;
  for (const auto& photon : *photonHandle) {
    index++;

    reco::PhotonRef photonref(photonHandle, index);
    photons.push_back(photon);
    auto& newPhoton = photons.back();

    if ((applyPhotonCalibOnData_ && event.isRealData()) || (applyPhotonCalibOnMC_ && !event.isRealData())) {
      calibratePhoton(newPhoton, photonref, *photonCalibEnergyHandle, *photonCalibEnergyErrHandle);
    }

    //we do this after calibration
    bool keep = keepPhotonSel_(newPhoton);
    if (!keep) {
      photons.pop_back();
      continue;
    }

    //fill pf candidate value map vector
    pfCandIsoPairVecPho.push_back((*photonPfCandMapHandle)[photonref]);

    //fill photon id valuemap vectors
    int subindex = 0;
    for (const auto& photonIdHandle : photonIdHandles) {
      photonIdVals[subindex++].push_back((*photonIdHandle)[photonref]);
    }

    subindex = 0;
    for (const auto& photonFloatValueMapHandle : photonFloatValueMapHandles) {
      photonFloatValueMapVals[subindex++].push_back((*photonFloatValueMapHandle)[photonref]);
    }

    // HI photon isolation
    if (!recoHIPhotonIsolationMapInputToken_.isUninitialized()) {
      recoHIPhotonIsolationMapInputVals.push_back((*recoHIPhotonIsolationMapInputValueMap)[photonref]);
    }

    //link photon core
    const reco::PhotonCoreRef& photonCore = photon.photonCore();
    linkCore(photonCore, photonCores, photonCoreMap);

    bool slimRelink = slimRelinkPhotonSel_(newPhoton);
    //no supercluster relinking unless slimRelink selection is satisfied
    if (!slimRelink)
      continue;

    bool relink = relinkPhotonSel_(newPhoton);

    //link supercluster
    const reco::SuperClusterRef& superCluster = photon.superCluster();
    linkSuperCluster(superCluster, superClusterMap, superClusters, relink, superClusterFullRelinkMap);

    //conversions only for full relinking
    if (!relink)
      continue;

    const reco::ConversionRefVector& convrefs = photon.conversions();
    linkConversions(convrefs, conversions, conversionMap);

    //explicitly references conversions
    const reco::ConversionRefVector& singleconvrefs = photon.conversionsOneLeg();
    linkConversions(singleconvrefs, singleConversions, singleConversionMap);

    //hcal hits
    linkHcalHits(*photon.superCluster(), *hbheHitHandle, hcalRechitMap);
  }

  //loop over oot photons and fill maps
  //special note1: since not PFCand --> no PF isolation, IDs (but we do have FloatValueMap!)
  //special note2: conversion sequence not run over bcs from oot phos, so skip relinking of oot phos
  //special note3: clusters and superclusters in own collections!
  if (!ootPhotonT_.isUninitialized()) {
    index = -1;
    for (const auto& ootPhoton : *ootPhotonHandle) {
      index++;

      bool keep = keepOOTPhotonSel_(ootPhoton);
      if (!keep)
        continue;

      reco::PhotonRef ootPhotonref(ootPhotonHandle, index);

      ootPhotons.push_back(ootPhoton);

      //fill photon pfclusteriso valuemap vectors
      int subindex = 0;
      for (const auto& ootPhotonFloatValueMapHandle : ootPhotonFloatValueMapHandles) {
        ootPhotonFloatValueMapVals[subindex++].push_back((*ootPhotonFloatValueMapHandle)[ootPhotonref]);
      }

      //link photon core
      const reco::PhotonCoreRef& ootPhotonCore = ootPhoton.photonCore();
      linkCore(ootPhotonCore, ootPhotonCores, ootPhotonCoreMap);

      bool slimRelink = slimRelinkOOTPhotonSel_(ootPhoton);
      //no supercluster relinking unless slimRelink selection is satisfied
      if (!slimRelink)
        continue;

      bool relink = relinkOOTPhotonSel_(ootPhoton);

      const reco::SuperClusterRef& ootSuperCluster = ootPhoton.superCluster();
      linkSuperCluster(ootSuperCluster, ootSuperClusterMap, ootSuperClusters, relink, ootSuperClusterFullRelinkMap);
      //hcal hits
      linkHcalHits(*ootPhoton.superCluster(), *hbheHitHandle, hcalRechitMap);
    }
  }

  //loop over electrons and fill maps
  index = -1;
  for (const auto& gsfElectron : *gsfElectronHandle) {
    index++;

    reco::GsfElectronRef gsfElectronref(gsfElectronHandle, index);
    gsfElectrons.push_back(gsfElectron);
    auto& newGsfElectron = gsfElectrons.back();
    if ((applyGsfElectronCalibOnData_ && event.isRealData()) || (applyGsfElectronCalibOnMC_ && !event.isRealData())) {
      calibrateElectron(newGsfElectron,
                        gsfElectronref,
                        *gsfElectronCalibEnergyHandle,
                        *gsfElectronCalibEnergyErrHandle,
                        *gsfElectronCalibEcalEnergyHandle,
                        *gsfElectronCalibEcalEnergyErrHandle);
    }

    bool keep = keepGsfElectronSel_(newGsfElectron);
    if (!keep) {
      gsfElectrons.pop_back();
      continue;
    }

    pfCandIsoPairVecEle.push_back((*gsfElectronPfCandMapHandle)[gsfElectronref]);

    //fill electron id valuemap vectors
    int subindex = 0;
    for (const auto& gsfElectronIdHandle : gsfElectronIdHandles) {
      gsfElectronIdVals[subindex++].push_back((*gsfElectronIdHandle)[gsfElectronref]);
    }

    subindex = 0;
    for (const auto& gsfElectronFloatValueMapHandle : gsfElectronFloatValueMapHandles) {
      gsfElectronFloatValueMapVals[subindex++].push_back((*gsfElectronFloatValueMapHandle)[gsfElectronref]);
    }

    const reco::GsfElectronCoreRef& gsfElectronCore = gsfElectron.core();
    linkCore(gsfElectronCore, gsfElectronCores, gsfElectronCoreMap);

    const reco::GsfTrackRef& gsfTrack = gsfElectron.gsfTrack();

    // Save the main gsfTrack
    if (!gsfTrackMap.count(gsfTrack)) {
      gsfTracks.push_back(*gsfTrack);
      gsfTrackMap[gsfTrack] = gsfTracks.size() - 1;
    }

    // Save additional ambiguous gsf tracks in a map:
    for (auto const& ambigGsfTrack : gsfElectron.ambiguousGsfTracks()) {
      if (!gsfTrackMap.count(ambigGsfTrack)) {
        gsfTracks.push_back(*ambigGsfTrack);
        gsfTrackMap[ambigGsfTrack] = gsfTracks.size() - 1;
      }
    }

    bool slimRelink = slimRelinkGsfElectronSel_(newGsfElectron);
    //no supercluster relinking unless slimRelink selection is satisfied
    if (!slimRelink)
      continue;

    bool relink = relinkGsfElectronSel_(newGsfElectron);

    const reco::SuperClusterRef& superCluster = gsfElectron.superCluster();
    linkSuperCluster(superCluster, superClusterMap, superClusters, relink, superClusterFullRelinkMap);

    //conversions only for full relinking
    if (!relink)
      continue;

    const reco::ConversionRefVector& convrefs = gsfElectron.core()->conversions();
    linkConversions(convrefs, conversions, conversionMap);

    //explicitly references conversions
    const reco::ConversionRefVector& singleconvrefs = gsfElectron.core()->conversionsOneLeg();
    linkConversions(singleconvrefs, singleConversions, singleConversionMap);

    //conversions matched by trackrefs
    linkConversionsByTrackRef(conversionHandle, gsfElectron, conversions, conversionMap);

    //single leg conversions matched by trackrefs
    linkConversionsByTrackRef(singleConversionHandle, gsfElectron, singleConversions, singleConversionMap);

    //hcal hits
    linkHcalHits(*gsfElectron.superCluster(), *hbheHitHandle, hcalRechitMap);
  }

  //loop over output SuperClusters and fill maps
  index = 0;
  for (auto& superCluster : superClusters) {
    //link seed cluster no matter what
    const reco::CaloClusterPtr& seedCluster = superCluster.seed();
    linkCaloCluster(seedCluster, ebeeClusters, ebeeClusterMap);

    //only proceed if superCluster is marked for full relinking
    bool fullrelink = superClusterFullRelinkMap.count(index++);
    if (!fullrelink) {
      //zero detid vector which is anyways not useful without stored rechits
      superCluster.clearHitsAndFractions();
      continue;
    }

    // link calo clusters
    linkCaloClusters(superCluster,
                     ebeeClusters,
                     ebeeClusterMap,
                     rechitMap,
                     barrelHitHandle,
                     endcapHitHandle,
                     caloTopology,
                     esClusters,
                     esClusterMap);

    //conversions matched geometrically
    linkConversionsByTrackRef(conversionHandle, superCluster, conversions, conversionMap);

    //single leg conversions matched by trackrefs
    linkConversionsByTrackRef(singleConversionHandle, superCluster, singleConversions, singleConversionMap);
  }

  //loop over output OOTSuperClusters and fill maps
  if (!ootPhotonT_.isUninitialized()) {
    index = 0;
    for (auto& ootSuperCluster : ootSuperClusters) {
      //link seed cluster no matter what
      const reco::CaloClusterPtr& ootSeedCluster = ootSuperCluster.seed();
      linkCaloCluster(ootSeedCluster, ootEbeeClusters, ootEbeeClusterMap);

      //only proceed if ootSuperCluster is marked for full relinking
      bool fullrelink = ootSuperClusterFullRelinkMap.count(index++);
      if (!fullrelink) {
        //zero detid vector which is anyways not useful without stored rechits
        ootSuperCluster.clearHitsAndFractions();
        continue;
      }

      // link calo clusters
      linkCaloClusters(ootSuperCluster,
                       ootEbeeClusters,
                       ootEbeeClusterMap,
                       rechitMap,
                       barrelHitHandle,
                       endcapHitHandle,
                       caloTopology,
                       ootEsClusters,
                       ootEsClusterMap);
    }
  }
  //now finalize and add to the event collections in "reverse" order

  //rechits (fill output collections of rechits to be stored)
  for (const EcalRecHit& rechit : *barrelHitHandle) {
    if (rechitMap.count(rechit.detid())) {
      ebRecHits.push_back(rechit);
    }
  }

  for (const EcalRecHit& rechit : *endcapHitHandle) {
    if (rechitMap.count(rechit.detid())) {
      eeRecHits.push_back(rechit);
    }
  }

  event.emplace(outEBRecHits_, std::move(ebRecHits));
  event.emplace(outEERecHits_, std::move(eeRecHits));

  if (doPreshowerEcalHits_) {
    for (const EcalRecHit& rechit : *preshowerHitHandle) {
      if (rechitMap.count(rechit.detid())) {
        esRecHits.push_back(rechit);
      }
    }
    event.emplace(outESRecHits_, std::move(esRecHits));
  }

  for (const HBHERecHit& rechit : *hbheHitHandle) {
    if (hcalRechitMap.count(rechit.detid())) {
      hbheRecHits.push_back(rechit);
    }
  }
  event.emplace(outHBHERecHits_, std::move(hbheRecHits));

  //CaloClusters
  //put calocluster output collections in event and get orphan handles to create ptrs
  const auto& outEBEEClusterHandle = event.emplace(outEBEEClusters_, std::move(ebeeClusters));
  const auto& outESClusterHandle = event.emplace(outESClusters_, std::move(esClusters));
  ;

  //Loop over SuperClusters and relink GEDPhoton + GSFElectron CaloClusters
  for (reco::SuperCluster& superCluster : superClusters) {
    relinkCaloClusters(superCluster, ebeeClusterMap, esClusterMap, outEBEEClusterHandle, outESClusterHandle);
  }

  //OOTCaloClusters
  //put ootcalocluster output collections in event and get orphan handles to create ptrs
  edm::OrphanHandle<reco::CaloClusterCollection> outOOTEBEEClusterHandle;
  edm::OrphanHandle<reco::CaloClusterCollection> outOOTESClusterHandle;
  //Loop over OOTSuperClusters and relink OOTPhoton CaloClusters
  if (!ootPhotonT_.isUninitialized()) {
    outOOTEBEEClusterHandle = event.emplace(outOOTEBEEClusters_, std::move(ootEbeeClusters));
    outOOTESClusterHandle = event.emplace(outOOTESClusters_, std::move(ootEsClusters));
    for (reco::SuperCluster& ootSuperCluster : ootSuperClusters) {
      relinkCaloClusters(
          ootSuperCluster, ootEbeeClusterMap, ootEsClusterMap, outOOTEBEEClusterHandle, outOOTESClusterHandle);
    }
  }
  //put superclusters and conversions in the event
  const auto& outSuperClusterHandle = event.emplace(outSuperClusters_, std::move(superClusters));
  const auto& outConversionHandle = event.emplace(outConversions_, std::move(conversions));
  const auto& outSingleConversionHandle = event.emplace(outSingleConversions_, std::move(singleConversions));
  const auto& outGsfTrackHandle = event.emplace(outGsfTracks_, std::move(gsfTracks));

  //Loop over PhotonCores and relink GEDPhoton SuperClusters (and conversions)
  for (reco::PhotonCore& photonCore : photonCores) {
    // superclusters
    relinkSuperCluster(photonCore, superClusterMap, outSuperClusterHandle);

    //conversions
    const reco::ConversionRefVector& convrefs = photonCore.conversions();
    relinkConversions(photonCore, convrefs, conversionMap, outConversionHandle);

    //single leg conversions
    const reco::ConversionRefVector& singleconvrefs = photonCore.conversionsOneLeg();
    relinkConversions(photonCore, singleconvrefs, singleConversionMap, outSingleConversionHandle);
  }

  //Relink GSFElectron SuperClusters and main GSF Tracks
  for (reco::GsfElectronCore& gsfElectronCore : gsfElectronCores) {
    relinkSuperCluster(gsfElectronCore, superClusterMap, outSuperClusterHandle);
    relinkGsfTrack(gsfElectronCore, gsfTrackMap, outGsfTrackHandle);
  }

  //put ootsuperclusters in the event
  edm::OrphanHandle<reco::SuperClusterCollection> outOOTSuperClusterHandle;
  if (!ootPhotonT_.isUninitialized())
    outOOTSuperClusterHandle = event.emplace(outOOTSuperClusters_, std::move(ootSuperClusters));

  //Relink OOTPhoton SuperClusters
  for (reco::PhotonCore& ootPhotonCore : ootPhotonCores) {
    relinkSuperCluster(ootPhotonCore, ootSuperClusterMap, outOOTSuperClusterHandle);
  }

  //put photoncores and gsfelectroncores into the event
  const auto& outPhotonCoreHandle = event.emplace(outPhotonCores_, std::move(photonCores));
  edm::OrphanHandle<reco::PhotonCoreCollection> outOOTPhotonCoreHandle;
  if (!ootPhotonT_.isUninitialized())
    outOOTPhotonCoreHandle = event.emplace(outOOTPhotonCores_, std::move(ootPhotonCores));
  const auto& outgsfElectronCoreHandle = event.emplace(outGsfElectronCores_, std::move(gsfElectronCores));

  //loop over photons, oot photons, and electrons and relink the cores
  for (reco::Photon& photon : photons) {
    relinkPhotonCore(photon, photonCoreMap, outPhotonCoreHandle);
  }

  if (!ootPhotonT_.isUninitialized()) {
    for (reco::Photon& ootPhoton : ootPhotons) {
      relinkPhotonCore(ootPhoton, ootPhotonCoreMap, outOOTPhotonCoreHandle);
    }
  }

  for (reco::GsfElectron& gsfElectron : gsfElectrons) {
    relinkGsfElectronCore(gsfElectron, gsfElectronCoreMap, outgsfElectronCoreHandle);

    // -----
    // Also in this loop let's relink ambiguous tracks
    std::vector<reco::GsfTrackRef> ambigTracksInThisElectron;
    // Here we loop over the ambiguous tracks and save them in a vector
    for (auto const& igsf : gsfElectron.ambiguousGsfTracks()) {
      ambigTracksInThisElectron.push_back(igsf);
    }

    // Now we need to clear them (they are the refs to original collection):
    gsfElectron.clearAmbiguousGsfTracks();

    // And here we add them back, now from a new reduced collection:
    for (const auto& it : ambigTracksInThisElectron) {
      const auto& gsftkmapped = gsfTrackMap.find(it);

      if (gsftkmapped != gsfTrackMap.end()) {
        reco::GsfTrackRef gsftkref(outGsfTrackHandle, gsftkmapped->second);
        gsfElectron.addAmbiguousGsfTrack(gsftkref);
      } else
        throw cms::Exception("There must be a problem with linking and mapping of ambiguous gsf tracks...");
    }

    if (gsfElectron.ambiguousGsfTracksSize() > 0)
      gsfElectron.setAmbiguous(true);  // Set the flag

    ambigTracksInThisElectron.clear();
  }

  //(finally) store the output photon and electron collections
  const auto& outPhotonHandle = event.emplace(outPhotons_, std::move(photons));
  edm::OrphanHandle<reco::PhotonCollection> outOOTPhotonHandle;
  if (!ootPhotonT_.isUninitialized())
    outOOTPhotonHandle = event.emplace(outOOTPhotons_, std::move(ootPhotons));
  const auto& outGsfElectronHandle = event.emplace(outGsfElectrons_, std::move(gsfElectrons));

  //still need to output relinked valuemaps

  //photon pfcand isolation valuemap
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler fillerPhotons(photonPfCandMap);
  fillerPhotons.insert(outPhotonHandle, pfCandIsoPairVecPho.begin(), pfCandIsoPairVecPho.end());
  fillerPhotons.fill();

  //electron pfcand isolation valuemap
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler fillerGsfElectrons(gsfElectronPfCandMap);
  fillerGsfElectrons.insert(outGsfElectronHandle, pfCandIsoPairVecEle.begin(), pfCandIsoPairVecEle.end());
  fillerGsfElectrons.fill();

  event.emplace(outPhotonPfCandMap_, std::move(photonPfCandMap));
  event.emplace(outGsfElectronPfCandMap_, std::move(gsfElectronPfCandMap));

  //photon id value maps
  index = 0;
  for (auto const& vals : photonIdVals) {
    emplaceValueMap(outPhotonHandle, vals, event, outPhotonIds_[index++]);
  }

  //electron id value maps
  index = 0;
  for (auto const& vals : gsfElectronIdVals) {
    emplaceValueMap(outGsfElectronHandle, vals, event, outGsfElectronIds_[index++]);
  }

  // photon iso value maps
  index = 0;
  for (auto const& vals : photonFloatValueMapVals) {
    emplaceValueMap(outPhotonHandle, vals, event, outPhotonFloatValueMaps_[index++]);
  }

  if (!ootPhotonT_.isUninitialized()) {
    //oot photon iso value maps
    index = 0;
    for (auto const& vals : ootPhotonFloatValueMapVals) {
      emplaceValueMap(outOOTPhotonHandle, vals, event, outOOTPhotonFloatValueMaps_[index++]);
    }
  }

  // HI photon iso value maps
  if (!recoHIPhotonIsolationMapInputToken_.isUninitialized()) {
    emplaceValueMap(outPhotonHandle, recoHIPhotonIsolationMapInputVals, event, recoHIPhotonIsolationMapOutputName_);
  }

  //electron iso value maps
  index = 0;
  for (auto const& vals : gsfElectronFloatValueMapVals) {
    emplaceValueMap(outGsfElectronHandle, vals, event, outGsfElectronFloatValueMaps_[index++]);
  }

  event.put(outSlimmedSuperClusters_, std::move(produceSlimmedSuperClusters(event)));
}

template <typename T, typename U>
void ReducedEGProducer::linkCore(const T& core, U& cores, std::map<T, unsigned int>& coreMap) {
  if (!coreMap.count(core)) {
    cores.push_back(*core);
    coreMap[core] = cores.size() - 1;
  }
}

void ReducedEGProducer::linkSuperCluster(const reco::SuperClusterRef& superCluster,
                                         std::map<reco::SuperClusterRef, unsigned int>& superClusterMap,
                                         reco::SuperClusterCollection& superClusters,
                                         const bool relink,
                                         std::unordered_set<unsigned int>& superClusterFullRelinkMap) {
  const auto& mappedsc = superClusterMap.find(superCluster);
  //get index in output collection in order to keep track whether superCluster
  //will be subject to full relinking
  unsigned int mappedscidx = 0;
  if (mappedsc == superClusterMap.end()) {
    superClusters.push_back(*superCluster);
    mappedscidx = superClusters.size() - 1;
    superClusterMap[superCluster] = mappedscidx;
  } else {
    mappedscidx = mappedsc->second;
  }

  //additionally mark supercluster for full relinking
  if (relink)
    superClusterFullRelinkMap.insert(mappedscidx);
}

void ReducedEGProducer::linkConversions(const reco::ConversionRefVector& convrefs,
                                        reco::ConversionCollection& conversions,
                                        std::map<reco::ConversionRef, unsigned int>& conversionMap) {
  for (const auto& convref : convrefs) {
    linkConversion(convref, conversions, conversionMap);
  }
}

void ReducedEGProducer::linkConversionsByTrackRef(const edm::Handle<reco::ConversionCollection>& conversionHandle,
                                                  const reco::GsfElectron& gsfElectron,
                                                  reco::ConversionCollection& conversions,
                                                  std::map<reco::ConversionRef, unsigned int>& conversionMap) {
  int index = 0;
  for (const auto& conversion : *conversionHandle) {
    reco::ConversionRef convref(conversionHandle, index++);

    bool matched = ConversionTools::matchesConversion(gsfElectron, conversion, true, true);
    if (!matched)
      continue;

    linkConversion(convref, conversions, conversionMap);
  }
}

void ReducedEGProducer::linkConversionsByTrackRef(const edm::Handle<reco::ConversionCollection>& conversionHandle,
                                                  const reco::SuperCluster& superCluster,
                                                  reco::ConversionCollection& conversions,
                                                  std::map<reco::ConversionRef, unsigned int>& conversionMap) {
  int index = 0;
  for (const auto& conversion : *conversionHandle) {
    reco::ConversionRef convref(conversionHandle, index++);

    bool matched = ConversionTools::matchesConversion(superCluster, conversion, 0.2);
    if (!matched)
      continue;

    linkConversion(convref, conversions, conversionMap);
  }
}

void ReducedEGProducer::linkConversion(const reco::ConversionRef& convref,
                                       reco::ConversionCollection& conversions,
                                       std::map<reco::ConversionRef, unsigned int>& conversionMap) {
  if (!conversionMap.count(convref)) {
    conversions.push_back(*convref);
    conversionMap[convref] = conversions.size() - 1;
  }
}

void ReducedEGProducer::linkCaloCluster(const reco::CaloClusterPtr& caloCluster,
                                        reco::CaloClusterCollection& caloClusters,
                                        std::map<reco::CaloClusterPtr, unsigned int>& caloClusterMap) {
  if (!caloClusterMap.count(caloCluster)) {
    caloClusters.push_back(*caloCluster);
    caloClusterMap[caloCluster] = caloClusters.size() - 1;
  }
}

void ReducedEGProducer::linkCaloClusters(const reco::SuperCluster& superCluster,
                                         reco::CaloClusterCollection& ebeeClusters,
                                         std::map<reco::CaloClusterPtr, unsigned int>& ebeeClusterMap,
                                         std::unordered_set<DetId>& rechitMap,
                                         const edm::Handle<EcalRecHitCollection>& barrelHitHandle,
                                         const edm::Handle<EcalRecHitCollection>& endcapHitHandle,
                                         CaloTopology const& caloTopology,
                                         reco::CaloClusterCollection& esClusters,
                                         std::map<reco::CaloClusterPtr, unsigned int>& esClusterMap) {
  for (const auto& cluster : superCluster.clusters()) {
    linkCaloCluster(cluster, ebeeClusters, ebeeClusterMap);

    for (const auto& hitfrac : cluster->hitsAndFractions()) {
      rechitMap.insert(hitfrac.first);
    }
    //make sure to also take all hits in the 5x5 around the max energy xtal
    bool barrel = cluster->hitsAndFractions().front().first.subdetId() == EcalBarrel;
    const EcalRecHitCollection* rhcol = barrel ? barrelHitHandle.product() : endcapHitHandle.product();
    DetId seed = EcalClusterTools::getMaximum(*cluster, rhcol).first;

    std::vector<DetId> dets5x5 =
        caloTopology.getSubdetectorTopology(DetId::Ecal, barrel ? EcalBarrel : EcalEndcap)->getWindow(seed, 5, 5);
    for (const auto& detid : dets5x5) {
      rechitMap.insert(detid);
    }
  }
  for (const auto& cluster : superCluster.preshowerClusters()) {
    linkCaloCluster(cluster, esClusters, esClusterMap);

    for (const auto& hitfrac : cluster->hitsAndFractions()) {
      rechitMap.insert(hitfrac.first);
    }
  }
}

void ReducedEGProducer::linkHcalHits(const reco::SuperCluster& superClus,
                                     const HBHERecHitCollection& recHits,
                                     std::unordered_set<DetId>& hcalDetIds) {
  hcalHitSel_.addDetIds(superClus, recHits, hcalDetIds);
}

void ReducedEGProducer::relinkCaloClusters(reco::SuperCluster& superCluster,
                                           const std::map<reco::CaloClusterPtr, unsigned int>& ebeeClusterMap,
                                           const std::map<reco::CaloClusterPtr, unsigned int>& esClusterMap,
                                           const edm::OrphanHandle<reco::CaloClusterCollection>& outEBEEClusterHandle,
                                           const edm::OrphanHandle<reco::CaloClusterCollection>& outESClusterHandle) {
  //remap seed cluster
  const auto& seedmapped = ebeeClusterMap.find(superCluster.seed());
  if (seedmapped != ebeeClusterMap.end()) {
    //make new ptr
    reco::CaloClusterPtr clusptr(outEBEEClusterHandle, seedmapped->second);
    superCluster.setSeed(clusptr);
  }

  //remap all clusters
  reco::CaloClusterPtrVector clusters;
  for (const auto& cluster : superCluster.clusters()) {
    const auto& clustermapped = ebeeClusterMap.find(cluster);
    if (clustermapped != ebeeClusterMap.end()) {
      //make new ptr
      reco::CaloClusterPtr clusptr(outEBEEClusterHandle, clustermapped->second);
      clusters.push_back(clusptr);
    } else {
      //can only relink if all clusters are being relinked, so if one is missing, then skip the relinking completely
      clusters.clear();
      break;
    }
  }
  if (!clusters.empty()) {
    superCluster.setClusters(clusters);
  }

  //remap preshower clusters
  reco::CaloClusterPtrVector esclusters;
  for (const auto& cluster : superCluster.preshowerClusters()) {
    const auto& clustermapped = esClusterMap.find(cluster);
    if (clustermapped != esClusterMap.end()) {
      //make new ptr
      reco::CaloClusterPtr clusptr(outESClusterHandle, clustermapped->second);
      esclusters.push_back(clusptr);
    } else {
      //can only relink if all clusters are being relinked, so if one is missing, then skip the relinking completely
      esclusters.clear();
      break;
    }
  }
  if (!esclusters.empty()) {
    superCluster.setPreshowerClusters(esclusters);
  }
}

template <typename T>
void ReducedEGProducer::relinkSuperCluster(
    T& core,
    const std::map<reco::SuperClusterRef, unsigned int>& superClusterMap,
    const edm::OrphanHandle<reco::SuperClusterCollection>& outSuperClusterHandle) {
  const auto& scmapped = superClusterMap.find(core.superCluster());
  if (scmapped != superClusterMap.end()) {
    //make new ref
    reco::SuperClusterRef scref(outSuperClusterHandle, scmapped->second);
    core.setSuperCluster(scref);
  }
}

void ReducedEGProducer::relinkGsfTrack(reco::GsfElectronCore& gsfElectronCore,
                                       const std::map<reco::GsfTrackRef, unsigned int>& gsfTrackMap,
                                       const edm::OrphanHandle<reco::GsfTrackCollection>& outGsfTrackHandle) {
  const auto& gsftkmapped = gsfTrackMap.find(gsfElectronCore.gsfTrack());
  if (gsftkmapped != gsfTrackMap.end()) {
    reco::GsfTrackRef gsftkref(outGsfTrackHandle, gsftkmapped->second);
    gsfElectronCore.setGsfTrack(gsftkref);
  }
}

void ReducedEGProducer::relinkConversions(reco::PhotonCore& photonCore,
                                          const reco::ConversionRefVector& convrefs,
                                          const std::map<reco::ConversionRef, unsigned int>& conversionMap,
                                          const edm::OrphanHandle<reco::ConversionCollection>& outConversionHandle) {
  reco::ConversionRefVector outconvrefs;
  for (const auto& convref : convrefs) {
    const auto& convmapped = conversionMap.find(convref);
    if (convmapped != conversionMap.end()) {
      //make new ref
      reco::ConversionRef outref(outConversionHandle, convmapped->second);
    } else {
      //can only relink if all conversions are being relinked, so if one is missing, then skip the relinking completely
      outconvrefs.clear();
      break;
    }
  }
  if (!outconvrefs.empty()) {
    photonCore.setConversions(outconvrefs);
  }
}

void ReducedEGProducer::relinkPhotonCore(reco::Photon& photon,
                                         const std::map<reco::PhotonCoreRef, unsigned int>& photonCoreMap,
                                         const edm::OrphanHandle<reco::PhotonCoreCollection>& outPhotonCoreHandle) {
  const auto& coremapped = photonCoreMap.find(photon.photonCore());
  if (coremapped != photonCoreMap.end()) {
    //make new ref
    reco::PhotonCoreRef coreref(outPhotonCoreHandle, coremapped->second);
    photon.setPhotonCore(coreref);
  }
}

void ReducedEGProducer::relinkGsfElectronCore(
    reco::GsfElectron& gsfElectron,
    const std::map<reco::GsfElectronCoreRef, unsigned int>& gsfElectronCoreMap,
    const edm::OrphanHandle<reco::GsfElectronCoreCollection>& outgsfElectronCoreHandle) {
  const auto& coremapped = gsfElectronCoreMap.find(gsfElectron.core());
  if (coremapped != gsfElectronCoreMap.end()) {
    //make new ref
    reco::GsfElectronCoreRef coreref(outgsfElectronCoreHandle, coremapped->second);
    gsfElectron.setCore(coreref);
  }
}

void ReducedEGProducer::calibratePhoton(reco::Photon& photon,
                                        const reco::PhotonRef& oldPhoRef,
                                        const edm::ValueMap<float>& energyMap,
                                        const edm::ValueMap<float>& energyErrMap) {
  float newEnergy = energyMap[oldPhoRef];
  float newEnergyErr = energyErrMap[oldPhoRef];
  photon.setCorrectedEnergy(reco::Photon::P4type::regression2, newEnergy, newEnergyErr, true);
}

void ReducedEGProducer::calibrateElectron(reco::GsfElectron& electron,
                                          const reco::GsfElectronRef& oldEleRef,
                                          const edm::ValueMap<float>& energyMap,
                                          const edm::ValueMap<float>& energyErrMap,
                                          const edm::ValueMap<float>& ecalEnergyMap,
                                          const edm::ValueMap<float>& ecalEnergyErrMap) {
  const float newEnergy = energyMap[oldEleRef];
  const float newEnergyErr = energyErrMap[oldEleRef];
  const float newEcalEnergy = ecalEnergyMap[oldEleRef];
  const float newEcalEnergyErr = ecalEnergyErrMap[oldEleRef];

  //make a copy of this as the setCorrectedEcalEnergy call with modifiy the electrons p4
  const math::XYZTLorentzVector oldP4 = electron.p4();
  const float corr = newEnergy / oldP4.E();

  electron.setCorrectedEcalEnergy(newEcalEnergy);
  electron.setCorrectedEcalEnergyError(newEcalEnergyErr);

  math::XYZTLorentzVector newP4{oldP4.x() * corr, oldP4.y() * corr, oldP4.z() * corr, newEnergy};
  electron.correctMomentum(newP4, electron.trackMomentumError(), newEnergyErr);
}

std::unique_ptr<reco::SlimmedSuperClusterCollection> ReducedEGProducer::produceSlimmedSuperClusters(
    const edm::Event& event) const {
  auto outputSCs = std::make_unique<reco::SlimmedSuperClusterCollection>();
  for (const auto& inputSCToken : superClustersToSaveTs_) {
    auto inputSCs = event.get(inputSCToken);
    for (const auto& inputSC : inputSCs) {
      outputSCs->emplace_back(inputSC);
    }
  }

  auto tracks = event.get(trksForSCIso_);

  for (auto& sc : *outputSCs) {
    float trkIso = trkIsoCalc_.photonIsolation(sc.position(), &tracks).second;
    sc.setTrkIso(trkIso);
  }

  return outputSCs;
}
