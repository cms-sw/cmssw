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
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
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
#include "L1Trigger/TrackerDTC/interface/Setup.h"
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

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"
#include "L1Trigger/TrackTrigger/interface/TrackQuality.h"

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
    else {
      if (a.y() != b.y())
        return (b.y() > a.y());
      else
        return (a.z() > b.z());
    }
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
  trklet::Settings settings;

  // event processor for the tracklet track finding
  trklet::TrackletEventProcessor eventProcessor;

  unsigned int nHelixPar_;
  bool extended_;

  bool trackQuality_;
  std::unique_ptr<TrackQuality> trackQualityModel_;

  std::map<string, vector<int>> dtclayerdisk;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  edm::ESHandle<TrackerGeometry> tGeomHandle;

  edm::InputTag MCTruthClusterInputTag;
  edm::InputTag MCTruthStubInputTag;
  edm::InputTag TrackingParticleInputTag;

  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;

  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> ttClusterMCTruthToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> TrackingParticleToken_;
  edm::EDGetTokenT<TTDTC> tokenDTC_;

  // helper class to store DTC configuration
  trackerDTC::Setup setup_;

  // Setup token
  edm::ESGetToken<trackerDTC::Setup, trackerDTC::SetupRcd> esGetToken_;

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
      bsToken_(consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("BeamSpotSource"))),
      tokenDTC_(consumes<TTDTC>(edm::InputTag(iConfig.getParameter<edm::InputTag>("InputTagTTDTC")))) {
  if (readMoreMcTruth_) {
    ttClusterMCTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthClusterInputTag);
    TrackingParticleToken_ = consumes<std::vector<TrackingParticle>>(TrackingParticleInputTag);
  }

  produces<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>("Level1TTTracks").setBranchAlias("Level1TTTracks");

  asciiEventOutName_ = iConfig.getUntrackedParameter<string>("asciiFileName", "");

  fitPatternFile = iConfig.getParameter<edm::FileInPath>("fitPatternFile");
  processingModulesFile = iConfig.getParameter<edm::FileInPath>("processingModulesFile");
  memoryModulesFile = iConfig.getParameter<edm::FileInPath>("memoryModulesFile");
  wiresFile = iConfig.getParameter<edm::FileInPath>("wiresFile");

  extended_ = iConfig.getParameter<bool>("Extended");
  nHelixPar_ = iConfig.getParameter<unsigned int>("Hnpar");

  if (extended_) {
    tableTEDFile = iConfig.getParameter<edm::FileInPath>("tableTEDFile");
    tableTREFile = iConfig.getParameter<edm::FileInPath>("tableTREFile");
  }

  // book ES product
  esGetToken_ = esConsumes<trackerDTC::Setup, trackerDTC::SetupRcd, edm::Transition::BeginRun>();

  // --------------------------------------------------------------------------------
  // set options in Settings based on inputs from configuration files
  // --------------------------------------------------------------------------------

  settings.setExtended(extended_);
  settings.setNHelixPar(nHelixPar_);

  settings.setFitPatternFile(fitPatternFile.fullPath());
  settings.setProcessingModulesFile(processingModulesFile.fullPath());
  settings.setMemoryModulesFile(memoryModulesFile.fullPath());
  settings.setWiresFile(wiresFile.fullPath());

  if (extended_) {
    settings.setTableTEDFile(tableTEDFile.fullPath());
    settings.setTableTREFile(tableTREFile.fullPath());

    //FIXME: The TED and TRE tables are currently disabled by default, so we
    //need to allow for the additional tracklets that will eventually be
    //removed by these tables, once they are finalized
    settings.setNbitstrackletindex(10);
  }

  eventnum = 0;
  if (not asciiEventOutName_.empty()) {
    asciiEventOut_.open(asciiEventOutName_.c_str());
  }

  if (settings.debugTracklet()) {
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
    trackQualityModel_ = std::make_unique<TrackQuality>(iConfig.getParameter<edm::ParameterSet>("TrackQualityPSet"));
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
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0, 0, 0)).z();
  settings.setBfield(mMagneticFieldStrength);

  setup_ = iSetup.getData(esGetToken_);

  // initialize the tracklet event processing (this sets all the processing & memory modules, wiring, etc)
  eventProcessor.init(settings);
}

//////////
// PRODUCE
void L1FPGATrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  typedef std::map<trklet::L1TStub,
                   edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>,
                   L1TStubCompare>
      stubMapType;
  typedef edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>
      TTClusterRef;

  /// Prepare output
  auto L1TkTracksForOutput = std::make_unique<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>();

  stubMapType stubMap;

  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry> geometryHandle;

  /// Set pointers to Stacked Modules
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);

  ////////////
  // GET BS //
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(bsToken_, beamSpotHandle);
  math::XYZPoint bsPosition = beamSpotHandle->position();

  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);

  eventnum++;
  trklet::SLHCEvent ev;
  ev.setEventNum(eventnum);
  ev.setIP(bsPosition.x(), bsPosition.y());

  // tracking particles
  edm::Handle<std::vector<TrackingParticle>> TrackingParticleHandle;
  if (readMoreMcTruth_)
    iEvent.getByToken(TrackingParticleToken_, TrackingParticleHandle);

  // tracker topology
  const TrackerTopology* const tTopo = tTopoHandle.product();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle<TTDTC> handleDTC;
  iEvent.getByToken<TTDTC>(tokenDTC_, handleDTC);

  // must be defined for code to compile, even if it's not used unless readMoreMcTruth_ is true
  map<edm::Ptr<TrackingParticle>, int> translateTP;

  // MC truth association maps
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTClusterHandle;
  if (readMoreMcTruth_) {
    iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);

    ////////////////////////////////////////////////
    /// LOOP OVER TRACKING PARTICLES & GET SIMTRACKS

    int ntps = 1;  //count from 1 ; 0 will mean invalid

    int this_tp = 0;
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

  }  // end if (readMoreMcTruth_)

  /////////////////////////////////
  /// READ DTC STUB INFORMATION ///
  /////////////////////////////////

  // Process stubs in each region and channel within that region
  for (const int& region : handleDTC->tfpRegions()) {
    for (const int& channel : handleDTC->tfpChannels()) {
      // Get the DTC name form the channel

      static string dtcbasenames[12] = {
          "PS10G_1", "PS10G_2", "PS10G_3", "PS10G_4", "PS_1", "PS_2", "2S_1", "2S_2", "2S_3", "2S_4", "2S_5", "2S_6"};

      string dtcname = dtcbasenames[channel % 12];

      if (channel % 24 >= 12)
        dtcname = "neg" + dtcname;

      dtcname += (channel < 24) ? "_A" : "_B";

      // Get the stubs from the DTC
      const TTDTC::Stream& streamFromDTC{handleDTC->stream(region, channel)};

      // Prepare the DTC stubs for the IR
      for (size_t stubIndex = 0; stubIndex < streamFromDTC.size(); ++stubIndex) {
        const TTDTC::Frame& stub{streamFromDTC[stubIndex]};

        if (stub.first.isNull()) {
          continue;
        }

        const GlobalPoint& ttPos = setup_.stubPos(stub.first);

        //Get the 2 bits for the layercode
        string layerword = stub.second.to_string().substr(61, 2);
        unsigned int layercode = 2 * (layerword[0] - '0') + layerword[1] - '0';
        assert(layercode < 4);

        //translation from the two bit layercode to the layer/disk number of each of the
        //12 channels (dtcs)
        static int layerdisktab[12][4] = {{0, 6, 8, 10},
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

          const TTClusterRef& ttClusterRef = stub.first->clusterRef(iClus);

          // Now identify all TP's contributing to either cluster in stub.
          vector<edm::Ptr<TrackingParticle>> vecTpPtr = MCTruthTTClusterHandle->findTrackingParticlePtrs(ttClusterRef);

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

        double stubbend = stub.first->bendFE();  //stub.first->rawBend()
        if (ttPos.z() < -120) {
          stubbend = -stubbend;
        }

        ev.addStub(dtcname,
                   region,
                   layerdisk,
                   stubwordhex,
                   setup_.psModule(setup_.dtcId(region, channel)),
                   isFlipped,
                   ttPos.x(),
                   ttPos.y(),
                   ttPos.z(),
                   stubbend,
                   stub.first->innerClusterPosition(),
                   assocTPs);

        const trklet::L1TStub& lastStub = ev.lastStub();
        stubMap[lastStub] = stub.first;
      }
    }
  }

  //////////////////////////
  // NOW RUN THE L1 tracking

  if (!asciiEventOutName_.empty()) {
    ev.write(asciiEventOut_);
  }

  const std::vector<trklet::Track>& tracks = eventProcessor.tracks();

  // this performs the actual tracklet event processing
  eventProcessor.event(ev);

  int ntracks = 0;

  for (const auto& track : tracks) {
    if (track.duplicate())
      continue;

    ntracks++;

    // this is where we create the TTTrack object
    double tmp_rinv = track.rinv(settings);
    double tmp_phi = track.phi0(settings);
    double tmp_tanL = track.tanL(settings);
    double tmp_z0 = track.z0(settings);
    double tmp_d0 = track.d0(settings);
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
                                           settings.nHelixPar(),
                                           settings.bfield());

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

    stubMapType::const_iterator it;
    for (const auto& itstubs : stubs) {
      it = stubMap.find(itstubs);
      if (it != stubMap.end()) {
        aTrack.addStubRef(it->second);
      } else {
        // could not find stub in stub map
      }
    }

    // pt consistency
    aTrack.setStubPtConsistency(
        StubPtConsistency::getConsistency(aTrack, theTrackerGeom, tTopo, settings.bfield(), settings.nHelixPar()));

    // set TTTrack word
    aTrack.setTrackWordBits();

    if (trackQuality_) {
      trackQualityModel_->setTrackQuality(aTrack);
    }

    // test track word
    //aTrack.testTrackWordBits();

    L1TkTracksForOutput->push_back(aTrack);
  }

  iEvent.put(std::move(L1TkTracksForOutput), "Level1TTTracks");

}  /// End of produce()

// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1FPGATrackProducer);
