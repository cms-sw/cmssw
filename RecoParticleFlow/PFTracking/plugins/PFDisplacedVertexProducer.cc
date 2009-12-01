#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedVertexProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidateFwd.h"

#include <set>

using namespace std;
using namespace edm;

PFDisplacedVertexProducer::PFDisplacedVertexProducer(const edm::ParameterSet& iConfig) {
  
  // --- Setup input collection names --- //

  inputTagVertexCandidates_ 
    = iConfig.getParameter<InputTag>("vertexCandidates");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose");

  bool debug = 
    iConfig.getUntrackedParameter<bool>("debug");

  // ------ Algo Parameters ------ //

  // Maximal transverse distance between two minimal
  // approach points to be used together
  double transvSize 
     = iConfig.getParameter< double >("transvSize");
   
  // Maximal longitudinal distance between two minimal
  // approach points to be used together
  double longSize 
     = iConfig.getParameter< double >("longSize");

  // Minimal radius below which we do not reconstruct interactions
  // Typically the position of the first Pixel layer
  double primaryVertexCut
    = iConfig.getParameter< double >("primaryVertexCut");

  // Radius at which no secondary tracks are availables
  // in the barrel.For the moment we exclude the TOB barrel
  // since 5-th track step starts the latest at first TOB
  // layer.
  double tobCut 
     = iConfig.getParameter< double >("tobCut");

  // Radius at which no secondary tracks are availables
  // in the endcaps.For the moment we exclude the TEC wheel.
  double tecCut 
     = iConfig.getParameter< double >("tecCut");

  // The minimal accepted weight for the tracks calculated in the 
  // adaptive vertex fitter to be associated to the displaced vertex 
  double minAdaptWeight
    = iConfig.getParameter< double >("minAdaptWeight");

  produces<reco::PFDisplacedVertexCollection>();

  // Vertex Finder parameters  -----------------------------------
  pfDisplacedVertexFinder_.setDebug(debug);
  pfDisplacedVertexFinder_.setParameters(transvSize, longSize,  
					 primaryVertexCut, tobCut, 
					 tecCut, minAdaptWeight);
     
}



PFDisplacedVertexProducer::~PFDisplacedVertexProducer() { }



void 
PFDisplacedVertexProducer::beginJob() { }

void 
PFDisplacedVertexProducer::beginRun(edm::Run & run, 
			  const edm::EventSetup & es) { }


void 
PFDisplacedVertexProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) {
  
  LogDebug("PFDisplacedVertexProducer")<<"START event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;
  
  // Prepare useful information for the Finder

  ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);
  const MagneticField* theMagField = magField.product();

  ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

  ESHandle<TrackerGeometry> tkerGeomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(tkerGeomHandle);

  Handle<reco::PFDisplacedVertexCandidateCollection> vertexCandidates;
  iEvent.getByLabel(inputTagVertexCandidates_, vertexCandidates);
    
  // Fill useful event information for the Finder
  pfDisplacedVertexFinder_.setEdmParameters(theMagField, globTkGeomHandle, tkerGeomHandle); 
  pfDisplacedVertexFinder_.setInput(vertexCandidates);

  // Run the finder
  pfDisplacedVertexFinder_.findDisplacedVertices();
  

  if(verbose_) {
    ostringstream  str;
    str<<pfDisplacedVertexFinder_<<endl;
    cout << pfDisplacedVertexFinder_<<endl;
    LogInfo("PFDisplacedVertexProducer") << str.str()<<endl;
  }    


  auto_ptr< reco::PFDisplacedVertexCollection > 
    pOutputDisplacedVertexCollection( 
      pfDisplacedVertexFinder_.transferDisplacedVertices() ); 


  
  iEvent.put(pOutputDisplacedVertexCollection);
 
  LogDebug("PFDisplacedVertexProducer")<<"STOP event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;

}
