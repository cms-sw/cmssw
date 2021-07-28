/** \class ConversionTrackCandidateProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateCaloClusterAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"

#include <vector>

class ConversionTrackCandidateProducer : public edm::stream::EDProducer<> {
public:
  ConversionTrackCandidateProducer(const edm::ParameterSet& ps);

  void beginRun(edm::Run const&, edm::EventSetup const& es) final;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// Initialize EventSetup objects at each event
  void setEventSetup(const edm::EventSetup& es);

  std::string OutInTrackCandidateCollection_;
  std::string InOutTrackCandidateCollection_;

  std::string OutInTrackSCAssociationCollection_;
  std::string InOutTrackSCAssociationCollection_;

  edm::EDGetTokenT<edm::View<reco::CaloCluster>> bcBarrelCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster>> bcEndcapCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster>> scHybridBarrelProducer_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster>> scIslandEndcapProducer_;
  edm::EDGetTokenT<HBHERecHitCollection> hbheRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelecalCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapecalCollection_;
  edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrkEvtToken_;

  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  double isoConeR_;
  double isoInnerConeR_;
  double isoEtaSlice_;
  double isoEtMin_;
  double isoEMin_;
  bool vetoClusteredHits_;
  bool useNumXtals_;

  std::vector<int> flagsexclEB_;
  std::vector<int> flagsexclEE_;
  std::vector<int> severitiesexclEB_;
  std::vector<int> severitiesexclEE_;

  double ecalIsoCut_offset_;
  double ecalIsoCut_slope_;

  edm::ESHandle<CaloGeometry> theCaloGeom_;

  std::unique_ptr<BaseCkfTrajectoryBuilder> theTrajectoryBuilder_;

  OutInConversionSeedFinder outInSeedFinder_;
  OutInConversionTrackFinder outInTrackFinder_;
  InOutConversionSeedFinder inOutSeedFinder_;
  InOutConversionTrackFinder inOutTrackFinder_;

  std::vector<edm::Ptr<reco::CaloCluster>> caloPtrVecOutIn_;
  std::vector<edm::Ptr<reco::CaloCluster>> caloPtrVecInOut_;

  std::vector<edm::Ref<reco::SuperClusterCollection>> vecOfSCRefForOutIn;
  std::vector<edm::Ref<reco::SuperClusterCollection>> vecOfSCRefForInOut;

  std::unique_ptr<ElectronHcalHelper> hcalHelper_;

  void buildCollections(bool detector,
                        const edm::Handle<edm::View<reco::CaloCluster>>& scHandle,
                        const edm::Handle<edm::View<reco::CaloCluster>>& bcHandle,
                        const EcalRecHitCollection& ecalRecHits,
                        const EcalSeverityLevelAlgo* sevLev,
                        ElectronHcalHelper const& hcalHelper,
                        TrackCandidateCollection& outInTracks,
                        TrackCandidateCollection& inOutTracks,
                        std::vector<edm::Ptr<reco::CaloCluster>>& vecRecOI,
                        std::vector<edm::Ptr<reco::CaloCluster>>& vecRecIO);
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ConversionTrackCandidateProducer);

namespace {
  auto createBaseCkfTrajectoryBuilder(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
    return BaseCkfTrajectoryBuilderFactory::get()->create(pset.getParameter<std::string>("ComponentType"), pset, iC);
  }
}  // namespace

ConversionTrackCandidateProducer::ConversionTrackCandidateProducer(const edm::ParameterSet& config)
    : bcBarrelCollection_{consumes(config.getParameter<edm::InputTag>("bcBarrelCollection"))},
      bcEndcapCollection_{consumes(config.getParameter<edm::InputTag>("bcEndcapCollection"))},
      scHybridBarrelProducer_{consumes(config.getParameter<edm::InputTag>("scHybridBarrelProducer"))},
      scIslandEndcapProducer_{consumes(config.getParameter<edm::InputTag>("scIslandEndcapProducer"))},

      hbheRecHits_{consumes(config.getParameter<edm::InputTag>("hbheRecHits"))},
      barrelecalCollection_{consumes(config.getParameter<edm::InputTag>("barrelEcalRecHitCollection"))},
      endcapecalCollection_{consumes(config.getParameter<edm::InputTag>("endcapEcalRecHitCollection"))},
      measurementTrkEvtToken_{consumes(edm::InputTag("MeasurementTrackerEvent"))},

      theTrajectoryBuilder_(createBaseCkfTrajectoryBuilder(
          config.getParameter<edm::ParameterSet>("TrajectoryBuilderPSet"), consumesCollector())),
      outInSeedFinder_{config, consumesCollector()},
      outInTrackFinder_{config, theTrajectoryBuilder_.get()},
      inOutSeedFinder_{config, consumesCollector()},
      inOutTrackFinder_{config, theTrajectoryBuilder_.get()} {
  OutInTrackCandidateCollection_ = config.getParameter<std::string>("outInTrackCandidateCollection");
  InOutTrackCandidateCollection_ = config.getParameter<std::string>("inOutTrackCandidateCollection");

  OutInTrackSCAssociationCollection_ = config.getParameter<std::string>("outInTrackCandidateSCAssociationCollection");
  InOutTrackSCAssociationCollection_ = config.getParameter<std::string>("inOutTrackCandidateSCAssociationCollection");

  hOverEConeSize_ = config.getParameter<double>("hOverEConeSize");
  maxHOverE_ = config.getParameter<double>("maxHOverE");
  minSCEt_ = config.getParameter<double>("minSCEt");
  isoConeR_ = config.getParameter<double>("isoConeR");
  isoInnerConeR_ = config.getParameter<double>("isoInnerConeR");
  isoEtaSlice_ = config.getParameter<double>("isoEtaSlice");
  isoEtMin_ = config.getParameter<double>("isoEtMin");
  isoEMin_ = config.getParameter<double>("isoEMin");
  vetoClusteredHits_ = config.getParameter<bool>("vetoClusteredHits");
  useNumXtals_ = config.getParameter<bool>("useNumXstals");
  ecalIsoCut_offset_ = config.getParameter<double>("ecalIsoCut_offset");
  ecalIsoCut_slope_ = config.getParameter<double>("ecalIsoCut_slope");

  //Flags and Severities to be excluded from photon calculations
  auto const& flagnamesEB = config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEB");
  auto const& flagnamesEE = config.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEE");

  flagsexclEB_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);
  flagsexclEE_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  auto const& severitynamesEB = config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEB");
  auto const& severitynamesEE = config.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEE");

  severitiesexclEB_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);
  severitiesexclEE_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  // Register the product
  produces<TrackCandidateCollection>(OutInTrackCandidateCollection_);
  produces<TrackCandidateCollection>(InOutTrackCandidateCollection_);

  produces<reco::TrackCandidateCaloClusterPtrAssociation>(OutInTrackSCAssociationCollection_);
  produces<reco::TrackCandidateCaloClusterPtrAssociation>(InOutTrackSCAssociationCollection_);

  ElectronHcalHelper::Configuration cfgCone;
  cfgCone.hOverEConeSize = hOverEConeSize_;
  if (cfgCone.hOverEConeSize > 0) {
    cfgCone.onlyBehindCluster = false;
    cfgCone.checkHcalStatus = false;

    cfgCone.hbheRecHits = hbheRecHits_;

    cfgCone.eThresHB = config.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
    cfgCone.maxSeverityHB = config.getParameter<int>("maxHcalRecHitSeverity");
    cfgCone.eThresHE = config.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");
    cfgCone.maxSeverityHE = cfgCone.maxSeverityHB;
  }

  hcalHelper_ = std::make_unique<ElectronHcalHelper>(cfgCone, consumesCollector());
}

void ConversionTrackCandidateProducer::setEventSetup(const edm::EventSetup& theEventSetup) {
  outInSeedFinder_.setEventSetup(theEventSetup);
  inOutSeedFinder_.setEventSetup(theEventSetup);
  outInTrackFinder_.setEventSetup(theEventSetup);
  inOutTrackFinder_.setEventSetup(theEventSetup);
}

void ConversionTrackCandidateProducer::beginRun(edm::Run const& r, edm::EventSetup const& theEventSetup) {
  edm::ESHandle<NavigationSchool> nav;
  theEventSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  const NavigationSchool* navigation = nav.product();
  theTrajectoryBuilder_->setNavigationSchool(navigation);
  outInSeedFinder_.setNavigationSchool(navigation);
  inOutSeedFinder_.setNavigationSchool(navigation);
}

void ConversionTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  // get the trajectory builder and initialize it with the data
  theTrajectoryBuilder_->setEvent(theEvent, theEventSetup, &theEvent.get(measurementTrkEvtToken_));

  // this need to be done after the initialization of the TrajectoryBuilder!
  setEventSetup(theEventSetup);

  outInSeedFinder_.setEvent(theEvent);
  inOutSeedFinder_.setEvent(theEvent);

  //
  // create empty output collections
  //
  //  Out In Track Candidates
  auto outInTrackCandidate_p = std::make_unique<TrackCandidateCollection>();
  //  In Out  Track Candidates
  auto inOutTrackCandidate_p = std::make_unique<TrackCandidateCollection>();
  //   Track Candidate  calo  Cluster Association
  auto outInAssoc_p = std::make_unique<reco::TrackCandidateCaloClusterPtrAssociation>();
  auto inOutAssoc_p = std::make_unique<reco::TrackCandidateCaloClusterPtrAssociation>();

  // Get the basic cluster collection in the Barrel
  bool validBarrelBCHandle = true;
  auto bcBarrelHandle = theEvent.getHandle(bcBarrelCollection_);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConversionTrackCandidateProducer") << "Error! Can't get the Barrel Basic Clusters!";
    validBarrelBCHandle = false;
  }

  // Get the basic cluster collection in the Endcap
  bool validEndcapBCHandle = true;
  auto bcEndcapHandle = theEvent.getHandle(bcEndcapCollection_);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the Endcap Basic Clusters";
    validEndcapBCHandle = false;
  }

  // Get the Super Cluster collection in the Barrel
  bool validBarrelSCHandle = true;
  auto scBarrelHandle = theEvent.getHandle(scHybridBarrelProducer_);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the barrel superclusters!";
    validBarrelSCHandle = false;
  }

  // Get the Super Cluster collection in the Endcap
  bool validEndcapSCHandle = true;
  auto scEndcapHandle = theEvent.getHandle(scIslandEndcapProducer_);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the endcap superclusters!";
    validEndcapSCHandle = false;
  }

  // get the geometry from the event setup:
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeom_);

  hcalHelper_->beginEvent(theEvent, theEventSetup);

  auto const& ecalhitsCollEB = theEvent.get(barrelecalCollection_);
  auto const& ecalhitsCollEE = theEvent.get(endcapecalCollection_);

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  theEventSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();

  caloPtrVecOutIn_.clear();
  caloPtrVecInOut_.clear();

  bool isBarrel = true;
  if (validBarrelBCHandle && validBarrelSCHandle)
    buildCollections(isBarrel,
                     scBarrelHandle,
                     bcBarrelHandle,
                     ecalhitsCollEB,
                     sevLevel,
                     *hcalHelper_,
                     *outInTrackCandidate_p,
                     *inOutTrackCandidate_p,
                     caloPtrVecOutIn_,
                     caloPtrVecInOut_);

  if (validEndcapBCHandle && validEndcapSCHandle) {
    isBarrel = false;
    buildCollections(isBarrel,
                     scEndcapHandle,
                     bcEndcapHandle,
                     ecalhitsCollEE,
                     sevLevel,
                     *hcalHelper_,
                     *outInTrackCandidate_p,
                     *inOutTrackCandidate_p,
                     caloPtrVecOutIn_,
                     caloPtrVecInOut_);
  }

  //  std::cout  << "  ConversionTrackCandidateProducer  caloPtrVecOutIn_ size " <<  caloPtrVecOutIn_.size() << " caloPtrVecInOut_ size " << caloPtrVecInOut_.size()  << "\n";

  // put all products in the event
  // Barrel
  //std::cout  << "ConversionTrackCandidateProducer Putting in the event " << (*outInTrackCandidate_p).size() << " Out In track Candidates " << "\n";
  auto const refprodOutInTrackC = theEvent.put(std::move(outInTrackCandidate_p), OutInTrackCandidateCollection_);
  //std::cout  << "ConversionTrackCandidateProducer  refprodOutInTrackC size  " <<  (*(refprodOutInTrackC.product())).size()  <<  "\n";
  //
  //std::cout  << "ConversionTrackCandidateProducer Putting in the event  " << (*inOutTrackCandidate_p).size() << " In Out track Candidates " <<  "\n";
  auto const refprodInOutTrackC = theEvent.put(std::move(inOutTrackCandidate_p), InOutTrackCandidateCollection_);
  //std::cout  << "ConversionTrackCandidateProducer  refprodInOutTrackC size  " <<  (*(refprodInOutTrackC.product())).size()  <<  "\n";

  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerOI(*outInAssoc_p);
  fillerOI.insert(refprodOutInTrackC, caloPtrVecOutIn_.begin(), caloPtrVecOutIn_.end());
  fillerOI.fill();
  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerIO(*inOutAssoc_p);
  fillerIO.insert(refprodInOutTrackC, caloPtrVecInOut_.begin(), caloPtrVecInOut_.end());
  fillerIO.fill();

  // std::cout  << "ConversionTrackCandidateProducer Putting in the event   OutIn track - SC association: size  " <<  (*outInAssoc_p).size() << "\n";
  theEvent.put(std::move(outInAssoc_p), OutInTrackSCAssociationCollection_);

  // std::cout << "ConversionTrackCandidateProducer Putting in the event   InOut track - SC association: size  " <<  (*inOutAssoc_p).size() << "\n";
  theEvent.put(std::move(inOutAssoc_p), InOutTrackSCAssociationCollection_);

  outInSeedFinder_.clear();
  inOutSeedFinder_.clear();
}

void ConversionTrackCandidateProducer::buildCollections(bool isBarrel,
                                                        const edm::Handle<edm::View<reco::CaloCluster>>& scHandle,
                                                        const edm::Handle<edm::View<reco::CaloCluster>>& bcHandle,
                                                        EcalRecHitCollection const& ecalRecHits,
                                                        const EcalSeverityLevelAlgo* sevLevel,
                                                        ElectronHcalHelper const& hcalHelper,
                                                        TrackCandidateCollection& outInTrackCandidates,
                                                        TrackCandidateCollection& inOutTrackCandidates,
                                                        std::vector<edm::Ptr<reco::CaloCluster>>& vecRecOI,
                                                        std::vector<edm::Ptr<reco::CaloCluster>>& vecRecIO)

{
  //  Loop over SC in the barrel and reconstruct converted photons
  for (auto const& aClus : scHandle->ptrs()) {
    // preselection based in Et and H/E cut.
    if (aClus->energy() / cosh(aClus->eta()) <= minSCEt_)
      continue;
    if (aClus->eta() > 1.479 && aClus->eta() < 1.556)
      continue;

    const reco::CaloCluster* pClus = &(*aClus);
    const reco::SuperCluster* sc = dynamic_cast<const reco::SuperCluster*>(pClus);
    double scEt = sc->energy() / cosh(sc->eta());
    double HoE = hcalHelper.hcalESum(*sc, 0) / sc->energy();
    if (HoE >= maxHOverE_)
      continue;

    //// Apply also ecal isolation
    EgammaRecHitIsolation ecalIso(
        isoConeR_, isoInnerConeR_, isoEtaSlice_, isoEtMin_, isoEMin_, theCaloGeom_, ecalRecHits, sevLevel, DetId::Ecal);

    ecalIso.setVetoClustered(vetoClusteredHits_);
    ecalIso.setUseNumCrystals(useNumXtals_);
    if (isBarrel) {
      ecalIso.doFlagChecks(flagsexclEB_);
      ecalIso.doSeverityChecks(&ecalRecHits, severitiesexclEB_);
    } else {
      ecalIso.doFlagChecks(flagsexclEE_);
      ecalIso.doSeverityChecks(&ecalRecHits, severitiesexclEE_);
    }

    double ecalIsolation = ecalIso.getEtSum(sc);
    if (ecalIsolation > ecalIsoCut_offset_ + ecalIsoCut_slope_ * scEt)
      continue;

    // Now launch the seed finding
    outInSeedFinder_.setCandidate(pClus->energy(),
                                  GlobalPoint(pClus->position().x(), pClus->position().y(), pClus->position().z()));
    outInSeedFinder_.makeSeeds(bcHandle);

    std::vector<Trajectory> theOutInTracks = outInTrackFinder_.tracks(outInSeedFinder_.seeds(), outInTrackCandidates);

    inOutSeedFinder_.setCandidate(pClus->energy(),
                                  GlobalPoint(pClus->position().x(), pClus->position().y(), pClus->position().z()));
    inOutSeedFinder_.setTracks(theOutInTracks);
    inOutSeedFinder_.makeSeeds(bcHandle);

    std::vector<Trajectory> theInOutTracks = inOutTrackFinder_.tracks(inOutSeedFinder_.seeds(), inOutTrackCandidates);

    // Debug
    //   std::cout  << "ConversionTrackCandidateProducer  theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() <<  " Event pointer to out in track size barrel " << outInTrackCandidates.size() << " in out track size " << inOutTrackCandidates.size() <<   "\n";

    //////////// Fill vectors of Ref to SC to be used for the Track-SC association
    for (auto it = theOutInTracks.begin(); it != theOutInTracks.end(); ++it) {
      vecRecOI.push_back(aClus);
      //     std::cout  << "ConversionTrackCandidateProducer Barrel OutIn Tracks Number of hits " << (*it).foundHits() << "\n";
    }

    for (auto it = theInOutTracks.begin(); it != theInOutTracks.end(); ++it) {
      vecRecIO.push_back(aClus);
      //     std::cout  << "ConversionTrackCandidateProducer Barrel InOut Tracks Number of hits " << (*it).foundHits() << "\n";
    }
  }
}

void ConversionTrackCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // conversionTrackCandidates
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("bcBarrelCollection", {"particleFlowSuperClusterECAL", "particleFlowBasicClusterECALBarrel"});
  desc.add<edm::InputTag>("bcEndcapCollection", {"particleFlowSuperClusterECAL", "particleFlowBasicClusterECALEndcap"});
  desc.add<edm::InputTag>("scHybridBarrelProducer",
                          {"particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"});
  desc.add<edm::InputTag>("scIslandEndcapProducer",
                          {"particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower"});

  desc.add<std::string>("outInTrackCandidateSCAssociationCollection", "outInTrackCandidateSCAssociationCollection");
  desc.add<std::string>("inOutTrackCandidateSCAssociationCollection", "inOutTrackCandidateSCAssociationCollection");
  desc.add<std::string>("outInTrackCandidateCollection", "outInTracksFromConversions");
  desc.add<std::string>("inOutTrackCandidateCollection", "inOutTracksFromConversions");

  desc.add<edm::InputTag>("barrelEcalRecHitCollection", {"ecalRecHit", "EcalRecHitsEB"});
  desc.add<edm::InputTag>("endcapEcalRecHitCollection", {"ecalRecHit", "EcalRecHitsEE"});
  desc.add<std::string>("MeasurementTrackerName", "");
  desc.add<std::string>("OutInRedundantSeedCleaner", "CachingSeedCleanerBySharedInput");
  desc.add<std::string>("InOutRedundantSeedCleaner", "CachingSeedCleanerBySharedInput");
  desc.add<bool>("useHitsSplitting", false);
  desc.add<int>("maxNumOfSeedsOutIn", 50);
  desc.add<int>("maxNumOfSeedsInOut", 50);
  desc.add<double>("bcEtCut", 1.5);
  desc.add<double>("bcECut", 1.5);
  desc.add<bool>("useEtCut", true);

  desc.add<edm::InputTag>("hbheRecHits", {"hbhereco"});
  desc.add<std::vector<double>>("recHitEThresholdHB", {0., 0., 0., 0.});
  desc.add<std::vector<double>>("recHitEThresholdHE", {0., 0., 0., 0., 0., 0., 0.});
  desc.add<int>("maxHcalRecHitSeverity", 999999);

  desc.add<double>("minSCEt", 20.0);
  desc.add<double>("hOverEConeSize", 0.15);
  desc.add<double>("maxHOverE", 0.15);
  desc.add<double>("isoInnerConeR", 3.5);
  desc.add<double>("isoConeR", 0.4);
  desc.add<double>("isoEtaSlice", 2.5);
  desc.add<double>("isoEtMin", 0.0);
  desc.add<double>("isoEMin", 0.08);
  desc.add<bool>("vetoClusteredHits", false);
  desc.add<bool>("useNumXstals", true);
  desc.add<double>("ecalIsoCut_offset", 999999999);  // alternative value: 4.2
  desc.add<double>("ecalIsoCut_slope", 0.0);         // alternative value: 0.003

  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("RecHitSeverityToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEE", {});
  desc.add<std::vector<std::string>>("RecHitSeverityToBeExcludedEE", {});

  desc.add<double>("fractionShared", 0.5);
  desc.add<std::string>("TrajectoryBuilder", "TrajectoryBuilderForConversions");
  {
    edm::ParameterSetDescription psd0;
    psd0.setUnknown();
    desc.add<edm::ParameterSetDescription>("TrajectoryBuilderPSet", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("propagatorAlongTISE", "alongMomElePropagator");
    psd0.add<int>("numberMeasurementsForFit", 4);
    psd0.add<std::string>("propagatorOppositeTISE", "oppositeToMomElePropagator");
    desc.add<edm::ParameterSetDescription>("TransientInitialStateEstimatorParameters", psd0);
  }
  desc.add<bool>("allowSharedFirstHit", true);
  desc.add<double>("ValidHitBonus", 5.0);
  desc.add<double>("MissingHitPenalty", 20.0);

  descriptions.add("conversionTrackCandidatesDefault", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}
