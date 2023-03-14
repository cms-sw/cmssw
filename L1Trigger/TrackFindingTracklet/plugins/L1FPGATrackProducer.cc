//////////////////////////
//  Producer by Anders  //
//     and Emmanuele    //
//    july 2012 @ CU    //
//////////////////////////

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
//
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"
//
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
//
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

///////////////
// Tracklet emulation
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Sector.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletEventProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Residual.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubKiller.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubStreamData.h"
#include "L1Trigger/TrackFindingTracklet/interface/HitPatternHelper.h"

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"
#include "L1Trigger/TrackTrigger/interface/L1TrackQuality.h"

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

//////////////
// NAMESPACES
using namespace edm;
using namespace std;
using namespace tt;
using namespace trklet;

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

/////////////////////////////////////
// this class is needed to make a map
// between different types of stubs
struct L1TStubCompare {
public:
  bool operator()(const trklet::L1TStub& a, const trklet::L1TStub& b) const {
    if (a.x() != b.x())
      return (b.x() > a.x());
    else if (a.y() != b.y())
      return (b.y() > a.y());
    else if (a.z() != b.z())
      return (a.z() > b.z());
    else
      return a.bend() > b.bend();
  }
};

class L1FPGATrackProducer : public edm::one::EDProducer<edm::one::WatchRuns> {
public:
  /// Constructor/destructor
  explicit L1FPGATrackProducer(const edm::ParameterSet& iConfig);
  ~L1FPGATrackProducer() override;

private:
  int eventnum;

  /// Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  bool readMoreMcTruth_;

  /// File path for configuration files
  edm::FileInPath fitPatternFile;
  edm::FileInPath memoryModulesFile;
  edm::FileInPath processingModulesFile;
  edm::FileInPath wiresFile;

  edm::FileInPath tableTEDFile;
  edm::FileInPath tableTREFile;

  string asciiEventOutName_;
  std::ofstream asciiEventOut_;

  // settings containing various constants for the tracklet processing
  trklet::Settings settings_;

  // event processor for the tracklet track finding
  trklet::TrackletEventProcessor eventProcessor;

  // used to "kill" stubs from a selected area of the detector
  StubKiller* stubKiller_;
  int failScenario_;

  unsigned int nHelixPar_;
  bool extended_;
  bool reduced_;

  bool trackQuality_;
  std::unique_ptr<L1TrackQuality> trackQualityModel_;

  std::map<string, vector<int>> dtclayerdisk;

  edm::InputTag MCTruthClusterInputTag;
  edm::InputTag MCTruthStubInputTag;
  edm::InputTag TrackingParticleInputTag;

  const edm::EDGetTokenT<reco::BeamSpot> getTokenBS_;
  const edm::EDGetTokenT<TTDTC> getTokenDTC_;
  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> getTokenTTClusterMCTruth_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> getTokenTrackingParticle_;

  // ED output token for clock and bit accurate tracks
  const edm::EDPutTokenT<Streams> putTokenTracks_;
  // ED output token for clock and bit accurate stubs
  const edm::EDPutTokenT<StreamsStub> putTokenStubs_;
  // ChannelAssignment token
  const ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
  // helper class to assign tracks to channel
  const ChannelAssignment* channelAssignment_;

  // helper class to store DTC configuration
  const Setup* setup_;
  // helper class to store configuration needed by HitPatternHelper
  const hph::Setup* setupHPH_;

  // Setup token
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> esGetTokenBfield_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> esGetTokenTGeom_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> esGetTokenTTopo_;
  const edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetToken_;
  const edm::ESGetToken<hph::Setup, hph::SetupRcd> esGetTokenHPH_;

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
};

//////////////
// CONSTRUCTOR
L1FPGATrackProducer::L1FPGATrackProducer(edm::ParameterSet const& iConfig)
    : config(iConfig),
      readMoreMcTruth_(iConfig.getParameter<bool>("readMoreMcTruth")),
      MCTruthClusterInputTag(readMoreMcTruth_ ? config.getParameter<edm::InputTag>("MCTruthClusterInputTag")
                                              : edm::InputTag()),
      MCTruthStubInputTag(readMoreMcTruth_ ? config.getParameter<edm::InputTag>("MCTruthStubInputTag")
                                           : edm::InputTag()),
      TrackingParticleInputTag(readMoreMcTruth_ ? iConfig.getParameter<edm::InputTag>("TrackingParticleInputTag")
                                                : edm::InputTag()),
      // book ED products
      getTokenBS_(consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("BeamSpotSource"))),
      getTokenDTC_(consumes<TTDTC>(edm::InputTag(iConfig.getParameter<edm::InputTag>("InputTagTTDTC")))),
      // book ED output token for clock and bit accurate tracks
      putTokenTracks_(produces<Streams>("Level1TTTracks")),
      // book ED output token for clock and bit accurate stubs
      putTokenStubs_(produces<StreamsStub>("Level1TTTracks")),
      // book ES products
      esGetTokenChannelAssignment_(esConsumes<ChannelAssignment, ChannelAssignmentRcd, Transition::BeginRun>()),
      esGetTokenBfield_(esConsumes<edm::Transition::BeginRun>()),
      esGetTokenTGeom_(esConsumes()),
      esGetTokenTTopo_(esConsumes()),
      esGetToken_(esConsumes<tt::Setup, tt::SetupRcd, edm::Transition::BeginRun>()),
      esGetTokenHPH_(esConsumes<hph::Setup, hph::SetupRcd, edm::Transition::BeginRun>()) {
  if (readMoreMcTruth_) {
    getTokenTTClusterMCTruth_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthClusterInputTag);
    getTokenTrackingParticle_ = consumes<std::vector<TrackingParticle>>(TrackingParticleInputTag);
  }

  produces<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>("Level1TTTracks").setBranchAlias("Level1TTTracks");

  asciiEventOutName_ = iConfig.getUntrackedParameter<string>("asciiFileName", "");

  fitPatternFile = iConfig.getParameter<edm::FileInPath>("fitPatternFile");
  processingModulesFile = iConfig.getParameter<edm::FileInPath>("processingModulesFile");
  memoryModulesFile = iConfig.getParameter<edm::FileInPath>("memoryModulesFile");
  wiresFile = iConfig.getParameter<edm::FileInPath>("wiresFile");

  failScenario_ = iConfig.getUntrackedParameter<int>("FailScenario", 0);

  extended_ = iConfig.getParameter<bool>("Extended");
  reduced_ = iConfig.getParameter<bool>("Reduced");
  nHelixPar_ = iConfig.getParameter<unsigned int>("Hnpar");

  if (extended_) {
    tableTEDFile = iConfig.getParameter<edm::FileInPath>("tableTEDFile");
    tableTREFile = iConfig.getParameter<edm::FileInPath>("tableTREFile");
  }

  // initial ES products
  channelAssignment_ = nullptr;
  setup_ = nullptr;

  // --------------------------------------------------------------------------------
  // set options in Settings based on inputs from configuration files
  // --------------------------------------------------------------------------------

  settings_.setExtended(extended_);
  settings_.setReduced(reduced_);
  settings_.setNHelixPar(nHelixPar_);

  settings_.setFitPatternFile(fitPatternFile.fullPath());
  settings_.setProcessingModulesFile(processingModulesFile.fullPath());
  settings_.setMemoryModulesFile(memoryModulesFile.fullPath());
  settings_.setWiresFile(wiresFile.fullPath());

  settings_.setFakefit(iConfig.getParameter<bool>("Fakefit"));
  settings_.setStoreTrackBuilderOutput(iConfig.getParameter<bool>("StoreTrackBuilderOutput"));
  settings_.setRemovalType(iConfig.getParameter<string>("RemovalType"));
  settings_.setDoMultipleMatches(iConfig.getParameter<bool>("DoMultipleMatches"));

  if (extended_) {
    settings_.setTableTEDFile(tableTEDFile.fullPath());
    settings_.setTableTREFile(tableTREFile.fullPath());

    //FIXME: The TED and TRE tables are currently disabled by default, so we
    //need to allow for the additional tracklets that will eventually be
    //removed by these tables, once they are finalized
    settings_.setNbitstrackletindex(15);
  }

  eventnum = 0;
  if (not asciiEventOutName_.empty()) {
    asciiEventOut_.open(asciiEventOutName_.c_str());
  }

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "fit pattern :     " << fitPatternFile.fullPath()
                                 << "\n process modules : " << processingModulesFile.fullPath()
                                 << "\n memory modules :  " << memoryModulesFile.fullPath()
                                 << "\n wires          :  " << wiresFile.fullPath();
    if (extended_) {
      edm::LogVerbatim("Tracklet") << "table_TED    :  " << tableTEDFile.fullPath()
                                   << "\n table_TRE    :  " << tableTREFile.fullPath();
    }
  }

  trackQuality_ = iConfig.getParameter<bool>("TrackQuality");
  if (trackQuality_) {
    trackQualityModel_ = std::make_unique<L1TrackQuality>(iConfig.getParameter<edm::ParameterSet>("TrackQualityPSet"));
  }
  if (settings_.storeTrackBuilderOutput() && (settings_.doMultipleMatches() || !settings_.removalType().empty())) {
    cms::Exception exception("ConfigurationNotSupported.");
    exception.addContext("L1FPGATrackProducer::produce");
    if (settings_.doMultipleMatches())
      exception << "Storing of TrackBuilder output does not support doMultipleMatches.";
    if (!settings_.removalType().empty())
      exception << "Storing of TrackBuilder output does not support duplicate removal.";
    throw exception;
  }
}

/////////////
// DESTRUCTOR
L1FPGATrackProducer::~L1FPGATrackProducer() {
  if (asciiEventOut_.is_open()) {
    asciiEventOut_.close();
  }
}

///////END RUN
//
void L1FPGATrackProducer::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

////////////
// BEGIN JOB
void L1FPGATrackProducer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  ////////////////////////
  // GET MAGNETIC FIELD //
  const MagneticField* theMagneticField = &iSetup.getData(esGetTokenBfield_);
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0, 0, 0)).z();
  settings_.setBfield(mMagneticFieldStrength);

  setup_ = &iSetup.getData(esGetToken_);

  settings_.passSetup(setup_);

  setupHPH_ = &iSetup.getData(esGetTokenHPH_);
  // Tracklet pattern reco output channel info.
  channelAssignment_ = &iSetup.getData(esGetTokenChannelAssignment_);
  // initialize the tracklet event processing (this sets all the processing & memory modules, wiring, etc)
  eventProcessor.init(settings_, setup_);
}

//////////
// PRODUCE
void L1FPGATrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  typedef std::map<trklet::L1TStub,
                   edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>,
                   L1TStubCompare>
      stubMapType;
  typedef std::map<unsigned int,
                   edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
      stubIndexMapType;
  typedef edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>
      TTClusterRef;

  /// Prepare output
  auto L1TkTracksForOutput = std::make_unique<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>();

  stubMapType stubMap;
  stubIndexMapType stubIndexMap;

  ////////////
  // GET BS //
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(getTokenBS_, beamSpotHandle);
  math::XYZPoint bsPosition = beamSpotHandle->position();

  eventnum++;
  trklet::SLHCEvent ev;
  ev.setEventNum(eventnum);
  ev.setIP(bsPosition.x(), bsPosition.y());

  // tracking particles
  edm::Handle<std::vector<TrackingParticle>> TrackingParticleHandle;
  if (readMoreMcTruth_)
    iEvent.getByToken(getTokenTrackingParticle_, TrackingParticleHandle);

  // tracker topology
  const TrackerTopology* const tTopo = &iSetup.getData(esGetTokenTTopo_);
  const TrackerGeometry* const theTrackerGeom = &iSetup.getData(esGetTokenTGeom_);

  // check killing stubs for detector degradation studies
  // if failType = 0, StubKiller does not kill any modules
  int failType = 0;
  if (failScenario_ < 0 || failScenario_ > 9) {
    edm::LogVerbatim("Tracklet") << "Invalid fail scenario! Ignoring input";
  } else
    failType = failScenario_;

  stubKiller_ = new StubKiller();
  stubKiller_->initialise(failType, tTopo, theTrackerGeom);

  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle<TTDTC> handleDTC;
  iEvent.getByToken<TTDTC>(getTokenDTC_, handleDTC);

  // must be defined for code to compile, even if it's not used unless readMoreMcTruth_ is true
  map<edm::Ptr<TrackingParticle>, int> translateTP;

  // MC truth association maps
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTClusterHandle;
  if (readMoreMcTruth_) {
    iEvent.getByToken(getTokenTTClusterMCTruth_, MCTruthTTClusterHandle);

    ////////////////////////////////////////////////
    /// LOOP OVER TRACKING PARTICLES & GET SIMTRACKS

    int ntps = 1;  //count from 1 ; 0 will mean invalid

    int this_tp = 0;
    if (readMoreMcTruth_) {
      for (const auto& iterTP : *TrackingParticleHandle) {
        edm::Ptr<TrackingParticle> tp_ptr(TrackingParticleHandle, this_tp);
        this_tp++;

        // only keep TPs producing a cluster
        if (MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).empty())
          continue;

        if (iterTP.g4Tracks().empty()) {
          continue;
        }

        int sim_eventid = iterTP.g4Tracks().at(0).eventId().event();
        int sim_type = iterTP.pdgId();
        float sim_pt = iterTP.pt();
        float sim_eta = iterTP.eta();
        float sim_phi = iterTP.phi();

        float vx = iterTP.vertex().x();
        float vy = iterTP.vertex().y();
        float vz = iterTP.vertex().z();

        if (sim_pt < 1.0 || std::abs(vz) > 100.0 || hypot(vx, vy) > 50.0)
          continue;

        ev.addL1SimTrack(sim_eventid, ntps, sim_type, sim_pt, sim_eta, sim_phi, vx, vy, vz);

        translateTP[tp_ptr] = ntps;
        ntps++;

      }  //end loop over TPs
    }

  }  // end if (readMoreMcTruth_)

  /////////////////////////////////
  /// READ DTC STUB INFORMATION ///
  /////////////////////////////////

  // Process stubs in each region and channel within that tracking region
  unsigned int theStubIndex = 0;
  for (const int& region : handleDTC->tfpRegions()) {
    for (const int& channel : handleDTC->tfpChannels()) {
      // Get the DTC name & ID from the channel
      unsigned int atcaSlot = channel % 12;
      string dtcname = settings_.slotToDTCname(atcaSlot);
      if (channel % 24 >= 12)
        dtcname = "neg" + dtcname;
      dtcname += (channel < 24) ? "_A" : "_B";  // which detector region
      int dtcId = setup_->dtcId(region, channel);

      // Get the stubs from the DTC
      const tt::StreamStub& streamFromDTC{handleDTC->stream(region, channel)};

      // Prepare the DTC stubs for the IR
      for (size_t stubIndex = 0; stubIndex < streamFromDTC.size(); ++stubIndex) {
        const tt::FrameStub& stub{streamFromDTC[stubIndex]};
        const TTStubRef& stubRef = stub.first;

        if (stubRef.isNull())
          continue;

        const GlobalPoint& ttPos = setup_->stubPos(stubRef);

        //Get the 2 bits for the layercode
        string layerword = stub.second.to_string().substr(61, 2);
        unsigned int layercode = 2 * (layerword[0] - '0') + layerword[1] - '0';
        assert(layercode < 4);

        //translation from the two bit layercode to the layer/disk number of each of the
        //12 channels (dtcs)
        // FIX: take this from DTC cabling map.
        static const int layerdisktab[12][4] = {{0, 6, 8, 10},
                                                {0, 7, 9, -1},
                                                {1, 7, -1, -1},
                                                {6, 8, 10, -1},
                                                {2, 7, -1, -1},
                                                {2, 9, -1, -1},
                                                {3, 4, -1, -1},
                                                {4, -1, -1, -1},
                                                {5, -1, -1, -1},
                                                {5, 8, -1, -1},
                                                {6, 9, -1, -1},
                                                {7, 10, -1, -1}};

        int layerdisk = layerdisktab[channel % 12][layercode];
        assert(layerdisk != -1);

        //Get the 36 bit word - skip the lowest 3 buts (status and layer code)
        constexpr int DTCLinkWordSize = 64;
        constexpr int StubWordSize = 36;
        constexpr int LayerandStatusCodeSize = 3;
        string stubword =
            stub.second.to_string().substr(DTCLinkWordSize - StubWordSize - LayerandStatusCodeSize, StubWordSize);
        string stubwordhex = "";

        //Loop over the 9 words in the 36 bit stub word
        for (unsigned int i = 0; i < 9; i++) {
          bitset<4> bits(stubword.substr(i * 4, 4));
          ulong val = bits.to_ulong();
          stubwordhex += ((val < 10) ? ('0' + val) : ('A' + val - 10));
        }

        /// Get the Inner and Outer TTCluster
        edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>
            innerCluster = stub.first->clusterRef(0);
        edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>
            outerCluster = stub.first->clusterRef(1);

        // -----------------------------------------------------
        // check module orientation, if flipped, need to store that information for track fit
        // -----------------------------------------------------

        const DetId innerDetId = innerCluster->getDetId();
        const GeomDetUnit* det_inner = theTrackerGeom->idToDetUnit(innerDetId);
        const auto* theGeomDet_inner = dynamic_cast<const PixelGeomDetUnit*>(det_inner);
        const PixelTopology* topol_inner = dynamic_cast<const PixelTopology*>(&(theGeomDet_inner->specificTopology()));

        MeasurementPoint coords_inner = innerCluster->findAverageLocalCoordinatesCentered();
        LocalPoint clustlp_inner = topol_inner->localPosition(coords_inner);
        GlobalPoint posStub_inner = theGeomDet_inner->surface().toGlobal(clustlp_inner);

        const DetId outerDetId = outerCluster->getDetId();
        const GeomDetUnit* det_outer = theTrackerGeom->idToDetUnit(outerDetId);
        const auto* theGeomDet_outer = dynamic_cast<const PixelGeomDetUnit*>(det_outer);
        const PixelTopology* topol_outer = dynamic_cast<const PixelTopology*>(&(theGeomDet_outer->specificTopology()));

        MeasurementPoint coords_outer = outerCluster->findAverageLocalCoordinatesCentered();
        LocalPoint clustlp_outer = topol_outer->localPosition(coords_outer);
        GlobalPoint posStub_outer = theGeomDet_outer->surface().toGlobal(clustlp_outer);

        bool isFlipped = (posStub_outer.mag() < posStub_inner.mag());

        vector<int> assocTPs;

        for (unsigned int iClus = 0; iClus <= 1; iClus++) {  // Loop over both clusters that make up stub.

          const TTClusterRef& ttClusterRef = stubRef->clusterRef(iClus);

          // Now identify all TP's contributing to either cluster in stub.
          if (readMoreMcTruth_) {
            vector<edm::Ptr<TrackingParticle>> vecTpPtr =
                MCTruthTTClusterHandle->findTrackingParticlePtrs(ttClusterRef);

            for (const edm::Ptr<TrackingParticle>& tpPtr : vecTpPtr) {
              if (translateTP.find(tpPtr) != translateTP.end()) {
                if (iClus == 0) {
                  assocTPs.push_back(translateTP.at(tpPtr));
                } else {
                  assocTPs.push_back(-translateTP.at(tpPtr));
                }
                // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
              } else {
                assocTPs.push_back(0);
              }
            }
          }
        }

        double stubbend = stubRef->bendFE();  //stubRef->rawBend()
        if (ttPos.z() < -120) {
          stubbend = -stubbend;
        }

        bool barrel = (layerdisk < N_LAYER);
        // See  https://github.com/cms-sw/cmssw/tree/master/Geometry/TrackerNumberingBuilder
        enum TypeBarrel { nonBarrel = 0, tiltedMinus = 1, tiltedPlus = 2, flat = 3 };
        const TypeBarrel type = static_cast<TypeBarrel>(tTopo->tobSide(innerDetId));
        bool tiltedBarrel = barrel && (type == tiltedMinus || type == tiltedPlus);
        unsigned int tiltedRingId = 0;
        // Tilted module ring no. (Increasing 1 to 12 as |z| increases).
        if (tiltedBarrel) {
          tiltedRingId = tTopo->tobRod(innerDetId);
          if (type == tiltedMinus) {
            unsigned int layp1 = 1 + layerdisk;  // Setup counts from 1
            unsigned int nTilted = setup_->numTiltedLayerRing(layp1);
            tiltedRingId = 1 + nTilted - tiltedRingId;
          }
        }
        // Endcap module ring number (1-15) in endcap disks.
        unsigned int endcapRingId = barrel ? 0 : tTopo->tidRing(innerDetId);

        const unsigned int intDetId = innerDetId.rawId();

        // check killing stubs for detector degredation studies
        const TTStub<Ref_Phase2TrackerDigi_>* theStub = &(*stubRef);
        bool killThisStub = stubKiller_->killStub(theStub);
        if (!killThisStub) {
          ev.addStub(dtcname,
                     region,
                     layerdisk,
                     stubwordhex,
                     setup_->psModule(dtcId),
                     isFlipped,
                     tiltedBarrel,
                     tiltedRingId,
                     endcapRingId,
                     intDetId,
                     ttPos.x(),
                     ttPos.y(),
                     ttPos.z(),
                     stubbend,
                     stubRef->innerClusterPosition(),
                     assocTPs,
                     theStubIndex);

          const trklet::L1TStub& lastStub = ev.lastStub();
          stubMap[lastStub] = stubRef;
          stubIndexMap[lastStub.uniqueIndex()] = stub.first;
          theStubIndex++;
        }
      }
    }
  }

  //////////////////////////
  // NOW RUN THE L1 tracking

  if (!asciiEventOutName_.empty()) {
    ev.write(asciiEventOut_);
  }

  const std::vector<trklet::Track>& tracks = eventProcessor.tracks();

  // max number of projection layers
  const unsigned int maxNumProjectionLayers = channelAssignment_->maxNumProjectionLayers();
  // number of track channels
  const unsigned int numStreamsTrack = N_SECTOR * channelAssignment_->numChannelsTrack();
  // number of stub channels
  const unsigned int numStreamsStub = N_SECTOR * channelAssignment_->numChannelsStub();
  // number of seeding layers
  const unsigned int numSeedingLayers = channelAssignment_->numSeedingLayers();
  // max number of stub channel per track
  const unsigned int numStubChannel = maxNumProjectionLayers + numSeedingLayers;
  // number of stub channels if all seed types streams padded to have same number of stub channels (for coding simplicity)
  const unsigned int numStreamsStubRaw = numStreamsTrack * numStubChannel;

  // Streams formatted to allow this code to run outside CMSSW.
  vector<vector<string>> streamsTrackRaw(numStreamsTrack);
  vector<vector<StubStreamData>> streamsStubRaw(numStreamsStubRaw);

  // this performs the actual tracklet event processing
  eventProcessor.event(ev, streamsTrackRaw, streamsStubRaw);

  for (const auto& track : tracks) {
    if (track.duplicate())
      continue;

    // this is where we create the TTTrack object
    double tmp_rinv = track.rinv(settings_);
    double tmp_phi = track.phi0(settings_);
    double tmp_tanL = track.tanL(settings_);
    double tmp_z0 = track.z0(settings_);
    double tmp_d0 = track.d0(settings_);
    double tmp_chi2rphi = track.chisqrphi();
    double tmp_chi2rz = track.chisqrz();
    unsigned int tmp_hit = track.hitpattern();

    TTTrack<Ref_Phase2TrackerDigi_> aTrack(tmp_rinv,
                                           tmp_phi,
                                           tmp_tanL,
                                           tmp_z0,
                                           tmp_d0,
                                           tmp_chi2rphi,
                                           tmp_chi2rz,
                                           0,
                                           0,
                                           0,
                                           tmp_hit,
                                           settings_.nHelixPar(),
                                           settings_.bfield());

    unsigned int trksector = track.sector();
    unsigned int trkseed = (unsigned int)abs(track.seed());

    aTrack.setPhiSector(trksector);
    aTrack.setTrackSeedType(trkseed);

    const vector<trklet::L1TStub>& stubptrs = track.stubs();
    vector<trklet::L1TStub> stubs;

    stubs.reserve(stubptrs.size());
    for (const auto& stubptr : stubptrs) {
      stubs.push_back(stubptr);
    }

    int countStubs = 0;
    stubMapType::const_iterator it;
    stubIndexMapType::const_iterator itIndex;
    for (const auto& itstubs : stubs) {
      itIndex = stubIndexMap.find(itstubs.uniqueIndex());
      if (itIndex != stubIndexMap.end()) {
        aTrack.addStubRef(itIndex->second);
        countStubs = countStubs + 1;
      } else {
        // could not find stub in stub map
      }
    }

    // pt consistency
    aTrack.setStubPtConsistency(
        StubPtConsistency::getConsistency(aTrack, theTrackerGeom, tTopo, settings_.bfield(), settings_.nHelixPar()));

    // set TTTrack word
    aTrack.setTrackWordBits();

    if (trackQuality_) {
      trackQualityModel_->setL1TrackQuality(aTrack);
    }

    //    hph::HitPatternHelper hph(setupHPH_, tmp_hit, tmp_tanL, tmp_z0);
    //    if (trackQuality_) {
    //      trackQualityModel_->setBonusFeatures(hph.bonusFeatures());
    //    }

    // test track word
    //aTrack.testTrackWordBits();

    L1TkTracksForOutput->push_back(aTrack);
  }

  iEvent.put(std::move(L1TkTracksForOutput), "Level1TTTracks");

  // produce clock and bit accurate stream output tracks and stubs.
  // from end of tracklet pattern recognition.
  // Convertion here is from stream format that allows this code to run
  // outside CMSSW to the EDProduct one.
  Streams streamsTrack(numStreamsTrack);
  StreamsStub streamsStub(numStreamsStub);

  for (unsigned int chanTrk = 0; chanTrk < numStreamsTrack; chanTrk++) {
    for (unsigned int itk = 0; itk < streamsTrackRaw[chanTrk].size(); itk++) {
      std::string bitsTrk = streamsTrackRaw[chanTrk][itk];
      int iSeed = chanTrk % channelAssignment_->numChannelsTrack();  // seed type
      streamsTrack[chanTrk].emplace_back(bitsTrk);

      const unsigned int chanStubOffsetIn = chanTrk * numStubChannel;
      const unsigned int chanStubOffsetOut = channelAssignment_->offsetStub(chanTrk);
      const unsigned int numProjLayers = channelAssignment_->numProjectionLayers(iSeed);
      TTBV hitMap(0, numProjLayers + numSeedingLayers);
      // remove padding from stub stream
      for (unsigned int iproj = 0; iproj < numStubChannel; iproj++) {
        // FW current has one (perhaps invalid) stub per layer per track.
        const StubStreamData& stubdata = streamsStubRaw[chanStubOffsetIn + iproj][itk];
        const L1TStub& stub = stubdata.stub();
        if (!stubdata.valid())
          continue;
        const TTStubRef& ttStubRef = stubMap[stub];
        const int seedType = stubdata.iSeed();
        const int layerId = setup_->layerId(ttStubRef);
        const int channelId = channelAssignment_->channelId(seedType, layerId);
        hitMap.set(channelId);
        streamsStub[chanStubOffsetOut + channelId].emplace_back(ttStubRef, stubdata.dataBits());
      }
      for (int layerId : hitMap.ids(false)) {  // invalid stubs
        streamsStub[chanStubOffsetOut + layerId].emplace_back(tt::FrameStub());
      }
    }
  }

  iEvent.emplace(putTokenTracks_, std::move(streamsTrack));
  iEvent.emplace(putTokenStubs_, std::move(streamsStub));

}  /// End of produce()

// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1FPGATrackProducer);
