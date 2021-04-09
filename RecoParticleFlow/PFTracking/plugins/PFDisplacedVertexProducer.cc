#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidateFwd.h"

#include <set>

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
