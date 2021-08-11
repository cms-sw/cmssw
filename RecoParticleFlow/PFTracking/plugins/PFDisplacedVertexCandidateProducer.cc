/**\class PFDisplacedVertexCandidateProducer 
\brief Producer for DisplacedVertices 

This producer makes use of DisplacedVertexCandidateFinder. This Finder
loop recursively over reco::Tracks to find those which are linked 
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
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexCandidateFinder.h"

class PFDisplacedVertexCandidateProducer : public edm::stream::EDProducer<> {
public:
  explicit PFDisplacedVertexCandidateProducer(const edm::ParameterSet&);

  ~PFDisplacedVertexCandidateProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /// Reco Tracks used to spot the nuclear interactions
  edm::EDGetTokenT<reco::TrackCollection> inputTagTracks_;

  /// Input tag for main vertex to cut of dxy of secondary tracks
  edm::EDGetTokenT<reco::VertexCollection> inputTagMainVertex_;
  edm::EDGetTokenT<reco::BeamSpot> inputTagBeamSpot_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  /// verbose ?
  bool verbose_;

  /// Displaced Vertex Candidates finder
  PFDisplacedVertexCandidateFinder pfDisplacedVertexCandidateFinder_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFDisplacedVertexCandidateProducer);

using namespace std;
using namespace edm;

PFDisplacedVertexCandidateProducer::PFDisplacedVertexCandidateProducer(const edm::ParameterSet& iConfig)
    : magneticFieldToken_(esConsumes()) {
  // --- Setup input collection names --- //
  inputTagTracks_ = consumes<reco::TrackCollection>(iConfig.getParameter<InputTag>("trackCollection"));

  inputTagMainVertex_ = consumes<reco::VertexCollection>(iConfig.getParameter<InputTag>("mainVertexLabel"));

  inputTagBeamSpot_ = consumes<reco::BeamSpot>(iConfig.getParameter<InputTag>("offlineBeamSpotLabel"));

  verbose_ = iConfig.getUntrackedParameter<bool>("verbose");

  bool debug = iConfig.getUntrackedParameter<bool>("debug");

  // ------ Algo Parameters ------ //

  // Distance of minimal approach below which
  // two tracks are considered as linked together
  double dcaCut = iConfig.getParameter<double>("dcaCut");

  // Do not reconstruct vertices wich are
  // too close to the beam pipe
  double primaryVertexCut = iConfig.getParameter<double>("primaryVertexCut");

  //maximum distance between the DCA Point and the inner hit of the track
  double dcaPInnerHitCut = iConfig.getParameter<double>("dcaPInnerHitCut");

  edm::ParameterSet ps_trk = iConfig.getParameter<edm::ParameterSet>("tracksSelectorParameters");

  // Collection to be produced
  produces<reco::PFDisplacedVertexCandidateCollection>();

  // Vertex Finder parameters  -----------------------------------
  pfDisplacedVertexCandidateFinder_.setDebug(debug);
  pfDisplacedVertexCandidateFinder_.setParameters(dcaCut, primaryVertexCut, dcaPInnerHitCut, ps_trk);
}

PFDisplacedVertexCandidateProducer::~PFDisplacedVertexCandidateProducer() {}

void PFDisplacedVertexCandidateProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("PFDisplacedVertexCandidateProducer")
      << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;

  // Prepare and fill useful event information for the Finder
  auto const& theMagField = &iSetup.getData(magneticFieldToken_);

  Handle<reco::TrackCollection> trackCollection;
  iEvent.getByToken(inputTagTracks_, trackCollection);

  Handle<reco::VertexCollection> mainVertexHandle;
  iEvent.getByToken(inputTagMainVertex_, mainVertexHandle);

  Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(inputTagBeamSpot_, beamSpotHandle);

  pfDisplacedVertexCandidateFinder_.setPrimaryVertex(mainVertexHandle, beamSpotHandle);
  pfDisplacedVertexCandidateFinder_.setInput(trackCollection, theMagField);

  // Run the finder
  pfDisplacedVertexCandidateFinder_.findDisplacedVertexCandidates();

  if (verbose_) {
    ostringstream str;
    str << pfDisplacedVertexCandidateFinder_ << endl;
    cout << pfDisplacedVertexCandidateFinder_ << endl;
    LogInfo("PFDisplacedVertexCandidateProducer") << str.str() << endl;
  }

  std::unique_ptr<reco::PFDisplacedVertexCandidateCollection> pOutputDisplacedVertexCandidateCollection(
      pfDisplacedVertexCandidateFinder_.transferVertexCandidates());

  iEvent.put(std::move(pOutputDisplacedVertexCandidateCollection));

  LogDebug("PFDisplacedVertexCandidateProducer")
      << "STOP event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;
}
