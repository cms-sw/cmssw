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
#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
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
#include "L1Trigger/TrackFindingTracklet/interface/Cabling.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletEventProcessor.h"

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

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
  bool operator()(const trklet::L1TStub& x, const trklet::L1TStub& y) const {
    if (x.x() != y.x())
      return (y.x() > x.x());
    else {
      if (x.y() != y.y())
        return (y.y() > x.y());
      else
        return (x.z() > y.z());
    }
  }
};

class L1FPGATrackProducer : public edm::stream::EDProducer<> {
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

  edm::FileInPath DTCLinkFile;
  edm::FileInPath moduleCablingFile;
  edm::FileInPath DTCLinkLayerDiskFile;

  edm::FileInPath tableTEDFile;
  edm::FileInPath tableTREFile;
  
  string asciiEventOutName_;
  std::ofstream asciiEventOut_;

  string geometryType_;

  // settings containing various constants for the tracklet processing
  trklet::Settings settings;

  // event processor for the tracklet track finding
  trklet::TrackletEventProcessor eventProcessor;

  unsigned int nHelixPar_;
  bool extended_;

  std::map<string, vector<int>> dtclayerdisk;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  edm::ESHandle<TrackerGeometry> tGeomHandle;

  edm::InputTag MCTruthClusterInputTag;
  edm::InputTag MCTruthStubInputTag;
  edm::InputTag TrackingParticleInputTag;
  edm::InputTag TrackingVertexInputTag;
  edm::InputTag ttStubSrc_;
  edm::InputTag bsSrc_;

  const edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> ttStubToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;

  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> ttClusterMCTruthToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubMCTruthToken_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> TrackingParticleToken_;
  edm::EDGetTokenT<std::vector<TrackingVertex>> TrackingVertexToken_;

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
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
      TrackingVertexInputTag(readMoreMcTruth_ ? iConfig.getParameter<edm::InputTag>("TrackingVertexInputTag")
                                              : edm::InputTag()),
      ttStubSrc_(config.getParameter<edm::InputTag>("TTStubSource")),
      bsSrc_(config.getParameter<edm::InputTag>("BeamSpotSource")),

      ttStubToken_(consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(ttStubSrc_)),
      bsToken_(consumes<reco::BeamSpot>(bsSrc_)) {
  if (readMoreMcTruth_) {
    ttClusterMCTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthClusterInputTag);
    ttStubMCTruthToken_ = consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthStubInputTag);
    TrackingParticleToken_ = consumes<std::vector<TrackingParticle>>(TrackingParticleInputTag);
    TrackingVertexToken_ = consumes<std::vector<TrackingVertex>>(TrackingVertexInputTag);
  }

  produces<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>("Level1TTTracks").setBranchAlias("Level1TTTracks");

  asciiEventOutName_ = iConfig.getUntrackedParameter<string>("asciiFileName", "");

  fitPatternFile = iConfig.getParameter<edm::FileInPath>("fitPatternFile");
  processingModulesFile = iConfig.getParameter<edm::FileInPath>("processingModulesFile");
  memoryModulesFile = iConfig.getParameter<edm::FileInPath>("memoryModulesFile");
  wiresFile = iConfig.getParameter<edm::FileInPath>("wiresFile");

  DTCLinkFile = iConfig.getParameter<edm::FileInPath>("DTCLinkFile");
  moduleCablingFile = iConfig.getParameter<edm::FileInPath>("moduleCablingFile");
  DTCLinkLayerDiskFile = iConfig.getParameter<edm::FileInPath>("DTCLinkLayerDiskFile");
  
  extended_ = iConfig.getParameter<bool>("Extended");
  nHelixPar_ = iConfig.getParameter<unsigned int>("Hnpar");

  if (extended_) {
    tableTEDFile = iConfig.getParameter<edm::FileInPath>("tableTEDFile");
    tableTREFile = iConfig.getParameter<edm::FileInPath>("tableTREFile");
  }

  
  // --------------------------------------------------------------------------------
  // set options in Settings based on inputs from configuration files
  // --------------------------------------------------------------------------------

  settings.setExtended(extended_);
  settings.setNHelixPar(nHelixPar_);

  settings.setDTCLinkFile(DTCLinkFile.fullPath());
  settings.setModuleCablingFile(moduleCablingFile.fullPath());
  settings.setDTCLinkLayerDiskFile(DTCLinkLayerDiskFile.fullPath());
  settings.setFitPatternFile(fitPatternFile.fullPath());
  settings.setProcessingModulesFile(processingModulesFile.fullPath());
  settings.setMemoryModulesFile(memoryModulesFile.fullPath());
  settings.setWiresFile(wiresFile.fullPath());
  
  if (extended_) {
    settings.setTableTEDFile(tableTEDFile.fullPath());
    settings.setTableTREFile(tableTREFile.fullPath());
  }

  eventnum = 0;
  if (not asciiEventOutName_.empty()) {
    asciiEventOut_.open(asciiEventOutName_.c_str());
  }

  if (settings.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "cabling DTC links :     " << DTCLinkFile.fullPath()
                                 << "\n module cabling :     " << moduleCablingFile.fullPath()
                                 << "\n DTC link layer disk :     " << DTCLinkLayerDiskFile.fullPath()
                                 << "\n fit pattern :     " << fitPatternFile.fullPath()
                                 << "\n process modules : " << processingModulesFile.fullPath()
                                 << "\n memory modules :  " << memoryModulesFile.fullPath()
                                 << "\n wires          :  " << wiresFile.fullPath();
    if (extended_) {
      edm::LogVerbatim("Tracklet") << "table_TED    :  " << tableTEDFile.fullPath()
				   << "\n table_TRE    :  " << tableTREFile.fullPath();
    }
  }
}

/////////////
// DESTRUCTOR
L1FPGATrackProducer::~L1FPGATrackProducer() {
  if (asciiEventOut_.is_open()) {
    asciiEventOut_.close();
  }
}

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

  // initialize the tracklet event processing (this sets all the processing & memory modules, wiring, etc)
  eventProcessor.init(&settings);
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
  ev.setIPx(bsPosition.x());
  ev.setIPy(bsPosition.y());

  // tracking particles
  edm::Handle<std::vector<TrackingParticle>> TrackingParticleHandle;
  edm::Handle<std::vector<TrackingVertex>> TrackingVertexHandle;
  if (readMoreMcTruth_)
    iEvent.getByToken(TrackingParticleToken_, TrackingParticleHandle);
  if (readMoreMcTruth_)
    iEvent.getByToken(TrackingVertexToken_, TrackingVertexHandle);

  // tracker topology
  const TrackerTopology* const tTopo = tTopoHandle.product();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken(ttStubToken_, Phase2TrackerDigiTTStubHandle);

  // must be defined for code to compile, even if it's not used unless readMoreMcTruth_ is true
  map<edm::Ptr<TrackingParticle>, int> translateTP;

  // MC truth association maps
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTClusterHandle;
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  if (readMoreMcTruth_) {
    iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);
    iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);

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

  ////////////////////////////////
  /// COLLECT STUB INFORMATION ///
  ////////////////////////////////

  bool firstPS = true;
  bool first2S = true;

  for (const auto& gd : theTrackerGeom->dets()) {
    DetId detid = (*gd).geographicalId();
    if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
      continue;  // only run on OT
    if (!tTopo->isLower(detid))
      continue;                              // loop on the stacks: choose the lower arbitrarily
    DetId stackDetid = tTopo->stack(detid);  // Stub module detid

    if (Phase2TrackerDigiTTStubHandle->find(stackDetid) == Phase2TrackerDigiTTStubHandle->end())
      continue;

    // Get the DetSets of the Clusters
    edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> stubs = (*Phase2TrackerDigiTTStubHandle)[stackDetid];
    const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit(detid);
    const auto* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(det0);
    const PixelTopology* topol = dynamic_cast<const PixelTopology*>(&(theGeomDet->specificTopology()));

    bool isPSmodule = theTrackerGeom->getDetectorType(detid) == TrackerGeometry::ModuleType::Ph2PSP;

    // set constants that are common for all modules/stubs of a given type (PS vs 2S)
    if (isPSmodule && firstPS) {
      settings.setNStrips_PS(topol->nrows());
      settings.setStripPitch_PS(topol->pitch().first);
      settings.setStripLength_PS(topol->pitch().second);
      firstPS = false;
    }
    if (!isPSmodule && first2S) {
      settings.setNStrips_2S(topol->nrows());
      settings.setStripPitch_2S(topol->pitch().first);
      settings.setStripLength_2S(topol->pitch().second);
      first2S = false;
    }

    // loop over stubs
    for (auto stubIter = stubs.begin(); stubIter != stubs.end(); ++stubIter) {
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubPtr =
          edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, stubIter);

      vector<int> assocTPs;

      if (readMoreMcTruth_) {
        for (unsigned int iClus = 0; iClus <= 1; iClus++) {  // Loop over both clusters that make up stub.

          const TTClusterRef& ttClusterRef = tempStubPtr->clusterRef(iClus);

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
      }  // end if (readMoreMcTruth_)

      MeasurementPoint coords = tempStubPtr->clusterRef(0)->findAverageLocalCoordinatesCentered();
      LocalPoint clustlp = topol->localPosition(coords);
      GlobalPoint posStub = theGeomDet->surface().toGlobal(clustlp);

      int eventID = -1;

      if (readMoreMcTruth_) {
        edm::Ptr<TrackingParticle> my_tp = MCTruthTTStubHandle->findTrackingParticlePtr(tempStubPtr);
      }

      int layer = -999999;
      int ladder = -999999;
      int module = -999999;

      int strip = 460;

      if (detid.subdetId() == StripSubdetector::TOB) {
        layer = static_cast<int>(tTopo->layer(detid));
        module = static_cast<int>(tTopo->module(detid));
        ladder = static_cast<int>(tTopo->tobRod(detid));

        // https://github.com/cms-sw/cmssw/tree/master/Geometry/TrackerNumberingBuilder
        // tobSide = 1: ring- (tilted)
        // tobSide = 2: ring+ (tilted)
        // tobSide = 3: barrel (flat)
        enum TypeBarrel { nonBarrel = 0, tiltedMinus = 1, tiltedPlus = 2, flat = 3 };
        const TypeBarrel type = static_cast<TypeBarrel>(tTopo->tobSide(detid));

        // modules in the flat part of barrel are mounted on planks, while modules in tilted part are on rings
        // below, "module" is the module number in the z direction (from minus z to positive),
        // while "ladder" is the module number in the phi direction

        if (layer > 0 && layer <= (int)trklet::N_PSLAYER) {
          if (type == tiltedMinus) {
            module = static_cast<int>(tTopo->tobRod(detid));
            ladder = static_cast<int>(tTopo->module(detid));
          }
          if (type == tiltedPlus) {
            module =
                trklet::N_TILTED_RINGS + trklet::N_MOD_PLANK.at(layer - 1) + static_cast<int>(tTopo->tobRod(detid));
            ladder = static_cast<int>(tTopo->module(detid));
          }
          if (type == flat) {
            module = trklet::N_TILTED_RINGS + static_cast<int>(tTopo->module(detid));
          }
        }
      } else if (detid.subdetId() == StripSubdetector::TID) {
        layer = 1000 + static_cast<int>(tTopo->tidRing(detid));
        ladder = static_cast<int>(tTopo->module(detid));
        module = static_cast<int>(tTopo->tidWheel(detid));
      }

      /// Get the Inner and Outer TTCluster
      edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>> innerCluster =
          tempStubPtr->clusterRef(0);
      edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>> outerCluster =
          tempStubPtr->clusterRef(1);

      std::vector<int> irphi;
      std::vector<int> innerrows = innerCluster->getRows();
      irphi.reserve(innerrows.size());
      for (int innerrow : innerrows) {
        irphi.push_back(innerrow);
      }
      std::vector<int> outerrows = outerCluster->getRows();
      for (int outerrow : outerrows) {
        irphi.push_back(outerrow);
      }

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

      // -----------------------------------------------------
      // correct sign for stubs in negative endcap
      float stub_bend = tempStubPtr->bendFE();
      float stub_pt = -1;
      if (layer > 999 && posStub.z() < 0.0) {
        stub_bend = -stub_bend;
      }
      if (!irphi.empty()) {
        strip = irphi[0];
      }

      //if module FE inefficiencies are calculated, a stub is thrown out if rawBend > 100
      if ((tempStubPtr->rawBend() < 100.) && (ev.addStub(layer,
                                                         ladder,
                                                         module,
                                                         strip,
                                                         eventID,
                                                         assocTPs,
                                                         stub_pt,
                                                         stub_bend,
                                                         posStub.x(),
                                                         posStub.y(),
                                                         posStub.z(),
                                                         isPSmodule,
                                                         isFlipped))) {
        const trklet::L1TStub& lastStub = ev.lastStub();
        stubMap[lastStub] = tempStubPtr;
      }
    }
  }

  //////////////////////////
  // NOW RUN THE L1 tracking

  if (!asciiEventOutName_.empty()) {
    ev.write(asciiEventOut_);
  }

  std::vector<trklet::Track*>& tracks = eventProcessor.tracks();

  trklet::L1SimTrack simtrk(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  // this performs the actual tracklet event processing
  eventProcessor.event(ev);

  int ntracks = 0;

  for (auto track : tracks) {
    if (track->duplicate())
      continue;

    ntracks++;

    // this is where we create the TTTrack object
    double tmp_rinv = track->rinv(settings);
    double tmp_phi = track->phi0(settings);
    double tmp_tanL = track->tanL(settings);
    double tmp_z0 = track->z0(settings);
    double tmp_d0 = track->d0(settings);
    double tmp_chi2rphi = track->chisqrphi();
    double tmp_chi2rz = track->chisqrz();
    unsigned int tmp_hit = track->hitpattern();

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

    unsigned int trksector = track->sector();
    unsigned int trkseed = (unsigned int)abs(track->seed());

    aTrack.setPhiSector(trksector);
    aTrack.setTrackSeedType(trkseed);

    const vector<const trklet::L1TStub*>& stubptrs = track->stubs();
    vector<trklet::L1TStub> stubs;

    stubs.reserve(stubptrs.size());
    for (auto stubptr : stubptrs) {
      stubs.push_back(*stubptr);
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

    // test track word
    //aTrack.testTrackWordBits();

    L1TkTracksForOutput->push_back(aTrack);
  }

  iEvent.put(std::move(L1TkTracksForOutput), "Level1TTTracks");

}  /// End of produce()

// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1FPGATrackProducer);
