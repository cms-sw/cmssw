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

private:
  /// Collection of DisplacedVertex Candidates used as input for
  /// the Displaced VertexFinder.
  edm::EDGetTokenT<reco::PFDisplacedVertexCandidateCollection> inputTagVertexCandidates_;

  /// Input tag for main vertex to cut of dxy of secondary tracks

  edm::EDGetTokenT<reco::VertexCollection> inputTagMainVertex_;
  edm::EDGetTokenT<reco::BeamSpot> inputTagBeamSpot_;

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

using namespace std;
using namespace edm;

PFDisplacedVertexProducer::PFDisplacedVertexProducer(const edm::ParameterSet& iConfig)
    : magFieldToken_(esConsumes()),
      globTkGeomToken_(esConsumes()),
      tkerTopoToken_(esConsumes()),
      tkerGeomToken_(esConsumes()) {
  // --- Setup input collection names --- //

  inputTagVertexCandidates_ =
      consumes<reco::PFDisplacedVertexCandidateCollection>(iConfig.getParameter<InputTag>("vertexCandidatesLabel"));

  inputTagMainVertex_ = consumes<reco::VertexCollection>(iConfig.getParameter<InputTag>("mainVertexLabel"));

  inputTagBeamSpot_ = consumes<reco::BeamSpot>(iConfig.getParameter<InputTag>("offlineBeamSpotLabel"));

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
  LogDebug("PFDisplacedVertexProducer") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run()
                                        << endl;

  // Prepare useful information for the Finder

  auto const& theMagField = &iSetup.getData(magFieldToken_);
  auto const& globTkGeom = &iSetup.getData(globTkGeomToken_);
  auto const& tkerTopo = &iSetup.getData(tkerTopoToken_);
  auto const& tkerGeom = &iSetup.getData(tkerGeomToken_);

  Handle<reco::PFDisplacedVertexCandidateCollection> vertexCandidates;
  iEvent.getByToken(inputTagVertexCandidates_, vertexCandidates);

  Handle<reco::VertexCollection> mainVertexHandle;
  iEvent.getByToken(inputTagMainVertex_, mainVertexHandle);

  Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(inputTagBeamSpot_, beamSpotHandle);

  // Fill useful event information for the Finder
  pfDisplacedVertexFinder_.setEdmParameters(theMagField, globTkGeom, tkerTopo, tkerGeom);
  pfDisplacedVertexFinder_.setPrimaryVertex(mainVertexHandle, beamSpotHandle);
  pfDisplacedVertexFinder_.setInput(vertexCandidates);

  // Run the finder
  pfDisplacedVertexFinder_.findDisplacedVertices();

  if (verbose_) {
    ostringstream str;
    //str<<pfDisplacedVertexFinder_<<endl;
    cout << pfDisplacedVertexFinder_ << endl;
    LogInfo("PFDisplacedVertexProducer") << str.str() << endl;
  }

  std::unique_ptr<reco::PFDisplacedVertexCollection> pOutputDisplacedVertexCollection(
      pfDisplacedVertexFinder_.transferDisplacedVertices());

  iEvent.put(std::move(pOutputDisplacedVertexCollection));

  LogDebug("PFDisplacedVertexProducer") << "STOP event: " << iEvent.id().event() << " in run " << iEvent.id().run()
                                        << endl;
}
