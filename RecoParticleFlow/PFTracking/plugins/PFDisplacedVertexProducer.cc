/**\class PFDisplacedVertexProducer 
\brief Producer for DisplacedVertices 

This producer makes use of DisplacedVertexFinder. This Finder fit vertex candidates
out of the DisplacedVertexCandidates which contain all tracks linked 
together by the criterion which is by default the minimal approach distance. 

\author Maxime Gouzevitch
\date   November 2009
*/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexFinder.h"

class PFDisplacedVertexProducer : public edm::stream::EDProducer<> {
public:
  explicit PFDisplacedVertexProducer(const edm::ParameterSet&);

  ~PFDisplacedVertexProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// Collection of DisplacedVertex Candidates used as input for
  /// the Displaced VertexFinder.
  const edm::EDGetTokenT<reco::PFDisplacedVertexCandidateCollection> inputTagVertexCandidates_;

  /// Input tag for main vertex to cut of dxy of secondary tracks

  const edm::EDGetTokenT<reco::VertexCollection> inputTagMainVertex_;
  const edm::EDGetTokenT<reco::BeamSpot> inputTagBeamSpot_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> globTkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkerTopoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkerGeomToken_;

  /// verbose ?
  bool verbose_;

  /// Displaced Vertices finder
  PFDisplacedVertexFinder pfDisplacedVertexFinder_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFDisplacedVertexProducer);

void PFDisplacedVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCandidatesLabel", {"particleFlowDisplacedVertexCandidate"});
  // verbosity
  desc.addUntracked<bool>("verbose", false);
  // Debug flag
  desc.addUntracked<bool>("debug", false);
  // maximum transverse distance between two points to be used in Seed
  desc.add<double>("transvSize", 1.0);
  // maximum longitudinal distance between two points to be used in Seed
  desc.add<double>("longSize", 5);
  // minimal radius below which we do not reconstruct interactions
  // typically the position of the first Pixel layer or beam pipe
  desc.add<double>("primaryVertexCut", 1.8);
  // radius below which we don't want to reconstruct displaced
  // vertices
  desc.add<double>("tobCut", 100);
  // z below which we don't want to reconstruct displaced
  // vertices
  desc.add<double>("tecCut", 220);
  // the minimal accepted weight for the tracks calculated in the
  // adaptive vertex fitter to be associated to the displaced vertex
  // this correspond to the sigmacut of 6
  desc.add<double>("minAdaptWeight", 0.5);
  // this flag is designed to reduce the timing of the algorithm in the high pile-up conditions. 2 tracks
  // vertices are the most sensitives to the pile-ups.
  desc.addUntracked<bool>("switchOff2TrackVertex", true);
  // ------------ Paramemeters for the track selection ------------
  // Primary vertex information used for dxy calculation
  desc.add<edm::InputTag>("mainVertexLabel", {"offlinePrimaryVertices", ""});
  desc.add<edm::InputTag>("offlineBeamSpotLabel", {"offlineBeamSpot", ""});
  // Parameters used to apply cuts
  {
    edm::ParameterSetDescription pset;
    pset.add<bool>("bSelectTracks", true);
    // If a track is high purity it is always kept
    pset.add<std::string>("quality", "HighPurity");
    // Following cuts are applyed to non high purity tracks
    // nChi2_max and pt_min cuts are applyed to the primary and secondary tracks
    pset.add<double>("nChi2_max", 5.);
    pset.add<double>("pt_min", 0.2);
    // nChi2_min applyed only to primary tracks which may be short
    // remove fake pixel triplets
    pset.add<double>("nChi2_min", 0.5);
    // Cuts applyed to the secondary tracks long and displaced
    pset.add<double>("dxy_min", 0.2);
    pset.add<int>("nHits_min", 6);
    pset.add<int>("nOuterHits_max", 9);
    desc.add<edm::ParameterSetDescription>("tracksSelectorParameters", pset);
  }
  // ------------ Paramemeters for the vertex identification ------------
  {
    edm::ParameterSetDescription pset;
    pset.add<bool>("bIdentifyVertices", true);
    // Minimal sum pt of secondary tracks for displaced vertices.
    // Below this value we find either loopers splitted in two parts eiter
    // fake vertices in forward direction
    pset.add<double>("pt_min", 0.5);
    // Minimal pT and log10(P_primary/P_secondary) for primary track in kinks (Primary+Secondary)
    // which are not identifier as K-+ decays
    pset.add<double>("pt_kink_min", 3.0);
    pset.add<double>("logPrimSec_min", 0.0);
    // maximum absoluta value of eta for loopers
    pset.add<double>("looper_eta_max", 0.1);
    // Masses cuts for selections
    //                                       CVmin  K0min  K0max  K-min  K-max  Ldmin  Ldmax  Nuclmin_ee
    pset.add<std::vector<double>>("masses", {0.050, 0.485, 0.515, 0.480, 0.520, 1.107, 1.125, 0.200});
    // Angle between the primaryVertex-secondaryVertex direction and secondary tracks direction
    // this angle means that the final system shall propagate in the same direction than initial system
    //                                       all_max, CV and V0 max
    pset.add<std::vector<double>>("angles", {15, 15});
    desc.add<edm::ParameterSetDescription>("vertexIdentifierParameters", pset);
  }
  // Adaptive Vertex Fitter parameters identical to the default ones except sigmacut.
  // The default value is sigmacut = 3 too tight for displaced vertices
  // see CMS NOTE-2008/033 for more details
  {
    edm::ParameterSetDescription pset;
    pset.add<double>("sigmacut", 6.);
    pset.add<double>("Tini", 256.);
    pset.add<double>("ratio", 0.25);
    desc.add<edm::ParameterSetDescription>("avfParameters", pset);
  }
  descriptions.add("particleFlowDisplacedVertex", desc);
}

using namespace std;
using namespace edm;

PFDisplacedVertexProducer::PFDisplacedVertexProducer(const edm::ParameterSet& iConfig)
    : inputTagVertexCandidates_(consumes<reco::PFDisplacedVertexCandidateCollection>(
          iConfig.getParameter<InputTag>("vertexCandidatesLabel"))),
      inputTagMainVertex_(consumes<reco::VertexCollection>(iConfig.getParameter<InputTag>("mainVertexLabel"))),
      inputTagBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<InputTag>("offlineBeamSpotLabel"))),
      magFieldToken_(esConsumes()),
      globTkGeomToken_(esConsumes()),
      tkerTopoToken_(esConsumes()),
      tkerGeomToken_(esConsumes()) {
  // --- Setup input collection names --- //

  verbose_ = iConfig.getUntrackedParameter<bool>("verbose");

  bool debug = iConfig.getUntrackedParameter<bool>("debug");

  // ------ Algo Parameters ------ //

  // Maximal transverse distance between two minimal
  // approach points to be used together
  double transvSize = iConfig.getParameter<double>("transvSize");

  // Maximal longitudinal distance between two minimal
  // approach points to be used together
  double longSize = iConfig.getParameter<double>("longSize");

  // Minimal radius below which we do not reconstruct interactions
  // Typically the position of the first Pixel layer
  double primaryVertexCut = iConfig.getParameter<double>("primaryVertexCut");

  // Radius at which no secondary tracks are availables
  // in the barrel.For the moment we exclude the TOB barrel
  // since 5-th track step starts the latest at first TOB
  // layer.
  double tobCut = iConfig.getParameter<double>("tobCut");

  // Radius at which no secondary tracks are availables
  // in the endcaps.For the moment we exclude the TEC wheel.
  double tecCut = iConfig.getParameter<double>("tecCut");

  // The minimal accepted weight for the tracks calculated in the
  // adaptive vertex fitter to be associated to the displaced vertex
  double minAdaptWeight = iConfig.getParameter<double>("minAdaptWeight");

  bool switchOff2TrackVertex = iConfig.getUntrackedParameter<bool>("switchOff2TrackVertex");

  edm::ParameterSet ps_trk = iConfig.getParameter<edm::ParameterSet>("tracksSelectorParameters");
  edm::ParameterSet ps_vtx = iConfig.getParameter<edm::ParameterSet>("vertexIdentifierParameters");
  edm::ParameterSet ps_avf = iConfig.getParameter<edm::ParameterSet>("avfParameters");

  produces<reco::PFDisplacedVertexCollection>();

  // Vertex Finder parameters  -----------------------------------
  pfDisplacedVertexFinder_.setDebug(debug);
  pfDisplacedVertexFinder_.setParameters(
      transvSize, longSize, primaryVertexCut, tobCut, tecCut, minAdaptWeight, switchOff2TrackVertex);
  pfDisplacedVertexFinder_.setAVFParameters(ps_avf);
  pfDisplacedVertexFinder_.setTracksSelector(ps_trk);
  pfDisplacedVertexFinder_.setVertexIdentifier(ps_vtx);
}

PFDisplacedVertexProducer::~PFDisplacedVertexProducer() {}

void PFDisplacedVertexProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("PFDisplacedVertexProducer") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run();

  // Prepare useful information for the Finder

  auto const& theMagField = &iSetup.getData(magFieldToken_);
  auto const& globTkGeom = &iSetup.getData(globTkGeomToken_);
  auto const& tkerTopo = &iSetup.getData(tkerTopoToken_);
  auto const& tkerGeom = &iSetup.getData(tkerGeomToken_);

  auto const& vertexCandidates = iEvent.getHandle(inputTagVertexCandidates_);
  auto const& mainVertexHandle = iEvent.getHandle(inputTagMainVertex_);
  auto const& beamSpotHandle = iEvent.getHandle(inputTagBeamSpot_);

  // Fill useful event information for the Finder
  pfDisplacedVertexFinder_.setEdmParameters(theMagField, globTkGeom, tkerTopo, tkerGeom);
  pfDisplacedVertexFinder_.setPrimaryVertex(mainVertexHandle, beamSpotHandle);
  pfDisplacedVertexFinder_.setInput(vertexCandidates);

  // Run the finder
  pfDisplacedVertexFinder_.findDisplacedVertices();

  if (verbose_) {
    ostringstream str;
    str << pfDisplacedVertexFinder_;
    edm::LogInfo("PFDisplacedVertexProducer") << str.str();
  }

  std::unique_ptr<reco::PFDisplacedVertexCollection> pOutputDisplacedVertexCollection(
      pfDisplacedVertexFinder_.transferDisplacedVertices());

  iEvent.put(std::move(pOutputDisplacedVertexCollection));

  LogDebug("PFDisplacedVertexProducer") << "STOP event: " << iEvent.id().event() << " in run " << iEvent.id().run();
}
