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
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateCaloClusterAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
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
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include <vector>

// ConversionTrackCandidateProducer inherits from EDProducer, so it can be a module:
class ConversionTrackCandidateProducer : public edm::stream::EDProducer<> {
public:
  ConversionTrackCandidateProducer(const edm::ParameterSet& ps);
  ~ConversionTrackCandidateProducer() override;

  void beginRun(edm::Run const&, edm::EventSetup const& es) final;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  int nEvt_;

  /// Initialize EventSetup objects at each event
  void setEventSetup(const edm::EventSetup& es);

  std::string OutInTrackCandidateCollection_;
  std::string InOutTrackCandidateCollection_;

  std::string OutInTrackSuperClusterAssociationCollection_;
  std::string InOutTrackSuperClusterAssociationCollection_;

  edm::EDGetTokenT<edm::View<reco::CaloCluster> > bcBarrelCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > bcEndcapCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > scHybridBarrelProducer_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > scIslandEndcapProducer_;
  edm::EDGetTokenT<CaloTowerCollection> hcalTowers_;
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

  std::unique_ptr<OutInConversionSeedFinder> theOutInSeedFinder_;
  std::unique_ptr<OutInConversionTrackFinder> theOutInTrackFinder_;
  std::unique_ptr<InOutConversionSeedFinder> theInOutSeedFinder_;
  std::unique_ptr<InOutConversionTrackFinder> theInOutTrackFinder_;

  std::vector<edm::Ptr<reco::CaloCluster> > caloPtrVecOutIn_;
  std::vector<edm::Ptr<reco::CaloCluster> > caloPtrVecInOut_;

  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForOutIn;
  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForInOut;

  void buildCollections(bool detector,
                        const edm::Handle<edm::View<reco::CaloCluster> >& scHandle,
                        const edm::Handle<edm::View<reco::CaloCluster> >& bcHandle,
                        edm::Handle<EcalRecHitCollection> ecalRecHitHandle,
                        const EcalRecHitCollection& ecalRecHits,
                        const EcalSeverityLevelAlgo* sevLev,
                        //edm::ESHandle<EcalChannelStatus>  chStatus,
                        const edm::Handle<CaloTowerCollection>& hcalTowersHandle,
                        TrackCandidateCollection& outInTracks,
                        TrackCandidateCollection& inOutTracks,
                        std::vector<edm::Ptr<reco::CaloCluster> >& vecRecOI,
                        std::vector<edm::Ptr<reco::CaloCluster> >& vecRecIO);
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ConversionTrackCandidateProducer);

namespace {
  auto createBaseCkfTrajectoryBuilder(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
    return BaseCkfTrajectoryBuilderFactory::get()->create(pset.getParameter<std::string>("ComponentType"), pset, iC);
  }
}  // namespace

ConversionTrackCandidateProducer::ConversionTrackCandidateProducer(const edm::ParameterSet& config)
    : theTrajectoryBuilder_(createBaseCkfTrajectoryBuilder(
          config.getParameter<edm::ParameterSet>("TrajectoryBuilderPSet"), consumesCollector())),
      theOutInSeedFinder_(new OutInConversionSeedFinder(config, consumesCollector())),
      theOutInTrackFinder_(new OutInConversionTrackFinder(config, theTrajectoryBuilder_.get())),
      theInOutSeedFinder_(new InOutConversionSeedFinder(config, consumesCollector())),
      theInOutTrackFinder_(new InOutConversionTrackFinder(config, theTrajectoryBuilder_.get())) {
  //std::cout << "ConversionTrackCandidateProducer CTOR " << "\n";
  nEvt_ = 0;

  // use onfiguration file to setup input/output collection names

  bcBarrelCollection_ =
      consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("bcBarrelCollection"));
  bcEndcapCollection_ =
      consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("bcEndcapCollection"));

  scHybridBarrelProducer_ =
      consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("scHybridBarrelProducer"));
  scIslandEndcapProducer_ =
      consumes<edm::View<reco::CaloCluster> >(config.getParameter<edm::InputTag>("scIslandEndcapProducer"));

  OutInTrackCandidateCollection_ = config.getParameter<std::string>("outInTrackCandidateCollection");
  InOutTrackCandidateCollection_ = config.getParameter<std::string>("inOutTrackCandidateCollection");

  OutInTrackSuperClusterAssociationCollection_ =
      config.getParameter<std::string>("outInTrackCandidateSCAssociationCollection");
  InOutTrackSuperClusterAssociationCollection_ =
      config.getParameter<std::string>("inOutTrackCandidateSCAssociationCollection");

  barrelecalCollection_ =
      consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("barrelEcalRecHitCollection"));
  endcapecalCollection_ =
      consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("endcapEcalRecHitCollection"));
  hcalTowers_ = consumes<CaloTowerCollection>(config.getParameter<edm::InputTag>("hcalTowers"));
  measurementTrkEvtToken_ = consumes<MeasurementTrackerEvent>(edm::InputTag("MeasurementTrackerEvent"));
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
  const std::vector<std::string> flagnamesEB =
      config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEB");

  const std::vector<std::string> flagnamesEE =
      config.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEE");

  flagsexclEB_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  flagsexclEE_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  const std::vector<std::string> severitynamesEB =
      config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEB");

  severitiesexclEB_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);

  const std::vector<std::string> severitynamesEE =
      config.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcludedEE");

  severitiesexclEE_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  // Register the product
  produces<TrackCandidateCollection>(OutInTrackCandidateCollection_);
  produces<TrackCandidateCollection>(InOutTrackCandidateCollection_);

  produces<reco::TrackCandidateCaloClusterPtrAssociation>(OutInTrackSuperClusterAssociationCollection_);
  produces<reco::TrackCandidateCaloClusterPtrAssociation>(InOutTrackSuperClusterAssociationCollection_);
}

ConversionTrackCandidateProducer::~ConversionTrackCandidateProducer() {}

void ConversionTrackCandidateProducer::setEventSetup(const edm::EventSetup& theEventSetup) {
  theOutInSeedFinder_->setEventSetup(theEventSetup);
  theInOutSeedFinder_->setEventSetup(theEventSetup);
  theOutInTrackFinder_->setEventSetup(theEventSetup);
  theInOutTrackFinder_->setEventSetup(theEventSetup);
}

void ConversionTrackCandidateProducer::beginRun(edm::Run const& r, edm::EventSetup const& theEventSetup) {
  edm::ESHandle<NavigationSchool> nav;
  theEventSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  const NavigationSchool* navigation = nav.product();
  theTrajectoryBuilder_->setNavigationSchool(navigation);
  theOutInSeedFinder_->setNavigationSchool(navigation);
  theInOutSeedFinder_->setNavigationSchool(navigation);
}

void ConversionTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  using namespace edm;
  nEvt_++;
  //  std::cout << "ConversionTrackCandidateProducer Analyzing event number " <<   theEvent.id() <<  " Global Counter " << nEvt_ << "\n";

  // get the trajectory builder and initialize it with the data
  edm::Handle<MeasurementTrackerEvent> data;
  theEvent.getByToken(measurementTrkEvtToken_, data);
  theTrajectoryBuilder_->setEvent(theEvent, theEventSetup, &*data);

  // this need to be done after the initialization of the TrajectoryBuilder!
  setEventSetup(theEventSetup);

  theOutInSeedFinder_->setEvent(theEvent);
  theInOutSeedFinder_->setEvent(theEvent);

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
  edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;
  theEvent.getByToken(bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConversionTrackCandidateProducer") << "Error! Can't get the Barrel Basic Clusters!";
    validBarrelBCHandle = false;
  }

  // Get the basic cluster collection in the Endcap
  bool validEndcapBCHandle = true;
  edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;
  theEvent.getByToken(bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the Endcap Basic Clusters";
    validEndcapBCHandle = false;
  }

  // Get the Super Cluster collection in the Barrel
  bool validBarrelSCHandle = true;
  edm::Handle<edm::View<reco::CaloCluster> > scBarrelHandle;
  theEvent.getByToken(scHybridBarrelProducer_, scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the barrel superclusters!";
    validBarrelSCHandle = false;
  }

  // Get the Super Cluster collection in the Endcap
  bool validEndcapSCHandle = true;
  edm::Handle<edm::View<reco::CaloCluster> > scEndcapHandle;
  theEvent.getByToken(scIslandEndcapProducer_, scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the endcap superclusters!";
    validEndcapSCHandle = false;
  }

  // get the geometry from the event setup:
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeom_);

  // get Hcal towers collection
  Handle<CaloTowerCollection> hcalTowersHandle;
  theEvent.getByToken(hcalTowers_, hcalTowersHandle);

  edm::Handle<EcalRecHitCollection> ecalhitsCollEB;
  edm::Handle<EcalRecHitCollection> ecalhitsCollEE;

  theEvent.getByToken(endcapecalCollection_, ecalhitsCollEE);
  theEvent.getByToken(barrelecalCollection_, ecalhitsCollEB);

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
                     *ecalhitsCollEB,
                     sevLevel,
                     hcalTowersHandle,
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
                     *ecalhitsCollEE,
                     sevLevel,
                     hcalTowersHandle,
                     *outInTrackCandidate_p,
                     *inOutTrackCandidate_p,
                     caloPtrVecOutIn_,
                     caloPtrVecInOut_);
  }

  //  std::cout  << "  ConversionTrackCandidateProducer  caloPtrVecOutIn_ size " <<  caloPtrVecOutIn_.size() << " caloPtrVecInOut_ size " << caloPtrVecInOut_.size()  << "\n";

  // put all products in the event
  // Barrel
  //std::cout  << "ConversionTrackCandidateProducer Putting in the event " << (*outInTrackCandidate_p).size() << " Out In track Candidates " << "\n";
  const edm::OrphanHandle<TrackCandidateCollection> refprodOutInTrackC =
      theEvent.put(std::move(outInTrackCandidate_p), OutInTrackCandidateCollection_);
  //std::cout  << "ConversionTrackCandidateProducer  refprodOutInTrackC size  " <<  (*(refprodOutInTrackC.product())).size()  <<  "\n";
  //
  //std::cout  << "ConversionTrackCandidateProducer Putting in the event  " << (*inOutTrackCandidate_p).size() << " In Out track Candidates " <<  "\n";
  const edm::OrphanHandle<TrackCandidateCollection> refprodInOutTrackC =
      theEvent.put(std::move(inOutTrackCandidate_p), InOutTrackCandidateCollection_);
  //std::cout  << "ConversionTrackCandidateProducer  refprodInOutTrackC size  " <<  (*(refprodInOutTrackC.product())).size()  <<  "\n";

  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerOI(*outInAssoc_p);
  fillerOI.insert(refprodOutInTrackC, caloPtrVecOutIn_.begin(), caloPtrVecOutIn_.end());
  fillerOI.fill();
  edm::ValueMap<reco::CaloClusterPtr>::Filler fillerIO(*inOutAssoc_p);
  fillerIO.insert(refprodInOutTrackC, caloPtrVecInOut_.begin(), caloPtrVecInOut_.end());
  fillerIO.fill();

  // std::cout  << "ConversionTrackCandidateProducer Putting in the event   OutIn track - SC association: size  " <<  (*outInAssoc_p).size() << "\n";
  theEvent.put(std::move(outInAssoc_p), OutInTrackSuperClusterAssociationCollection_);

  // std::cout << "ConversionTrackCandidateProducer Putting in the event   InOut track - SC association: size  " <<  (*inOutAssoc_p).size() << "\n";
  theEvent.put(std::move(inOutAssoc_p), InOutTrackSuperClusterAssociationCollection_);

  theOutInSeedFinder_->clear();
  theInOutSeedFinder_->clear();
}

void ConversionTrackCandidateProducer::buildCollections(bool isBarrel,
                                                        const edm::Handle<edm::View<reco::CaloCluster> >& scHandle,
                                                        const edm::Handle<edm::View<reco::CaloCluster> >& bcHandle,
                                                        edm::Handle<EcalRecHitCollection> ecalRecHitHandle,
                                                        const EcalRecHitCollection& ecalRecHits,
                                                        const EcalSeverityLevelAlgo* sevLevel,
                                                        //edm::ESHandle<EcalChannelStatus>  chStatus,
                                                        //const EcalChannelStatus* chStatus,
                                                        const edm::Handle<CaloTowerCollection>& hcalTowersHandle,
                                                        TrackCandidateCollection& outInTrackCandidates,
                                                        TrackCandidateCollection& inOutTrackCandidates,
                                                        std::vector<edm::Ptr<reco::CaloCluster> >& vecRecOI,
                                                        std::vector<edm::Ptr<reco::CaloCluster> >& vecRecIO)

{
  //std::cout << "ConversionTrackCandidateProducer is barrel " << isBarrel <<  "\n";
  //std::cout << "ConversionTrackCandidateProducer builcollections sc size " << scHandle->size() <<  "\n";
  //std::cout << "ConversionTrackCandidateProducer builcollections bc size " << bcHandle->size() <<  "\n";
  //const CaloGeometry* geometry = theCaloGeom_.product();

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
    const CaloTowerCollection* hcalTowersColl = hcalTowersHandle.product();
    EgammaTowerIsolation towerIso(hOverEConeSize_, 0., 0., -1, hcalTowersColl);
    double HoE = towerIso.getTowerESum(sc) / sc->energy();
    if (HoE >= maxHOverE_)
      continue;

    //// Apply also ecal isolation
    EgammaRecHitIsolation ecalIso(
        isoConeR_, isoInnerConeR_, isoEtaSlice_, isoEtMin_, isoEMin_, theCaloGeom_, ecalRecHits, sevLevel, DetId::Ecal);

    ecalIso.setVetoClustered(vetoClusteredHits_);
    ecalIso.setUseNumCrystals(useNumXtals_);
    if (isBarrel) {
      ecalIso.doFlagChecks(flagsexclEB_);
      ecalIso.doSeverityChecks(ecalRecHitHandle.product(), severitiesexclEB_);
    } else {
      ecalIso.doFlagChecks(flagsexclEE_);
      ecalIso.doSeverityChecks(ecalRecHitHandle.product(), severitiesexclEE_);
    }

    double ecalIsolation = ecalIso.getEtSum(sc);
    if (ecalIsolation > ecalIsoCut_offset_ + ecalIsoCut_slope_ * scEt)
      continue;

    // Now launch the seed finding
    theOutInSeedFinder_->setCandidate(pClus->energy(),
                                      GlobalPoint(pClus->position().x(), pClus->position().y(), pClus->position().z()));
    theOutInSeedFinder_->makeSeeds(bcHandle);

    std::vector<Trajectory> theOutInTracks =
        theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(), outInTrackCandidates);

    theInOutSeedFinder_->setCandidate(pClus->energy(),
                                      GlobalPoint(pClus->position().x(), pClus->position().y(), pClus->position().z()));
    theInOutSeedFinder_->setTracks(theOutInTracks);
    theInOutSeedFinder_->makeSeeds(bcHandle);

    std::vector<Trajectory> theInOutTracks =
        theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(), inOutTrackCandidates);

    // Debug
    //   std::cout  << "ConversionTrackCandidateProducer  theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() <<  " Event pointer to out in track size barrel " << outInTrackCandidates.size() << " in out track size " << inOutTrackCandidates.size() <<   "\n";

    //////////// Fill vectors of Ref to SC to be used for the Track-SC association
    for (std::vector<Trajectory>::const_iterator it = theOutInTracks.begin(); it != theOutInTracks.end(); ++it) {
      caloPtrVecOutIn_.push_back(aClus);
      //     std::cout  << "ConversionTrackCandidateProducer Barrel OutIn Tracks Number of hits " << (*it).foundHits() << "\n";
    }

    for (std::vector<Trajectory>::const_iterator it = theInOutTracks.begin(); it != theInOutTracks.end(); ++it) {
      caloPtrVecInOut_.push_back(aClus);
      //     std::cout  << "ConversionTrackCandidateProducer Barrel InOut Tracks Number of hits " << (*it).foundHits() << "\n";
    }
  }
}
