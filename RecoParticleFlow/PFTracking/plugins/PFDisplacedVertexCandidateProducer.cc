#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexCandidateProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <set>

using namespace std;
using namespace edm;

PFDisplacedVertexCandidateProducer::PFDisplacedVertexCandidateProducer(const edm::ParameterSet& iConfig) {
  
  // --- Setup input collection names --- //
  inputTagTracks_ 
    = iConfig.getParameter<InputTag>("trackCollection");

  inputTagMainVertex_ 
    = iConfig.getParameter<InputTag>("mainVertexLabel");

  inputTagBeamSpot_ 
    = iConfig.getParameter<InputTag>("offlineBeamSpotLabel");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose");

  bool debug = 
    iConfig.getUntrackedParameter<bool>("debug");

  // ------ Algo Parameters ------ //

  // Distance of minimal approach below which 
  // two tracks are considered as linked together
  double dcaCut 
    = iConfig.getParameter< double >("dcaCut");   

  // Do not reconstruct vertices wich are 
  // too close to the beam pipe
  double primaryVertexCut
    = iConfig.getParameter< double >("primaryVertexCut");   

  //maximum distance between the DCA Point and the inner hit of the track
  double dcaPInnerHitCut
    = iConfig.getParameter< double >("dcaPInnerHitCut");  

  edm::ParameterSet ps_trk 
    = iConfig.getParameter<edm::ParameterSet>("tracksSelectorParameters");

  // Collection to be produced
  produces<reco::PFDisplacedVertexCandidateCollection>();

  // Vertex Finder parameters  -----------------------------------
  pfDisplacedVertexCandidateFinder_.setDebug(debug);
  pfDisplacedVertexCandidateFinder_.setParameters(dcaCut, primaryVertexCut, dcaPInnerHitCut, ps_trk);
     
}


PFDisplacedVertexCandidateProducer::~PFDisplacedVertexCandidateProducer() { }



void 
PFDisplacedVertexCandidateProducer::beginJob() { }

void 
PFDisplacedVertexCandidateProducer::beginRun(edm::Run & run, 
			  const edm::EventSetup & es) { }


void 
PFDisplacedVertexCandidateProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) {
  
  LogDebug("PFDisplacedVertexCandidateProducer")<<"START event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;
  
  // Prepare and fill useful event information for the Finder
  edm::ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);
  const MagneticField* theMagField = magField.product();

  Handle <reco::TrackCollection> trackCollection;
  iEvent.getByLabel(inputTagTracks_, trackCollection);
    
  Handle< reco::VertexCollection > mainVertexHandle;
  iEvent.getByLabel(inputTagMainVertex_, mainVertexHandle);

  Handle< reco::BeamSpot > beamSpotHandle;
  iEvent.getByLabel(inputTagBeamSpot_, beamSpotHandle);

  pfDisplacedVertexCandidateFinder_.setPrimaryVertex(mainVertexHandle, beamSpotHandle);
  pfDisplacedVertexCandidateFinder_.setInput( trackCollection, theMagField );


  // Run the finder
  pfDisplacedVertexCandidateFinder_.findDisplacedVertexCandidates();
  

  if(verbose_) {
    ostringstream  str;
    str<<pfDisplacedVertexCandidateFinder_<<endl;
    cout << pfDisplacedVertexCandidateFinder_<<endl;
    LogInfo("PFDisplacedVertexCandidateProducer") << str.str()<<endl;
  }    


  auto_ptr< reco::PFDisplacedVertexCandidateCollection > 
    pOutputDisplacedVertexCandidateCollection(
      pfDisplacedVertexCandidateFinder_.transferVertexCandidates() ); 
  
  
  iEvent.put(pOutputDisplacedVertexCandidateCollection);
 
  LogDebug("PFDisplacedVertexCandidateProducer")<<"STOP event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;

}
