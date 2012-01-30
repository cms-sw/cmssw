#include "RecoTauTag/HLTProducers/interface/VertexFromTrackProducer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//using namespace reco;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
VertexFromTrackProducer::VertexFromTrackProducer(const edm::ParameterSet& conf)
  : theConfig(conf)
{
  edm::LogInfo("PVDebugInfo") 
    << "Initializing PV producer " << "\n";
  fVerbose=conf.getUntrackedParameter<bool>("verbose", false);
  trackLabel = conf.getParameter<edm::InputTag>("trackLabel");
  fIsRecoCandidate=conf.getParameter<bool>("isRecoCandidate");
  fUseBeamSpot=conf.getParameter<bool>("useBeamSpot");
  fUseVertex=conf.getParameter<bool>("useVertex");
  vertexLabel = conf.getParameter<edm::InputTag>("vertexLabel");
  beamSpotLabel = conf.getParameter<edm::InputTag>("beamSpotLabel");
 
  produces<reco::VertexCollection>();

}


VertexFromTrackProducer::~VertexFromTrackProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
VertexFromTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  std::auto_ptr<reco::VertexCollection> result(new reco::VertexCollection);
  reco::VertexCollection vColl;

  math::XYZPoint vertexPoint;

  if(fUseBeamSpot)
  {
  // get the BeamSpot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotLabel,recoBeamSpotHandle);
  if (recoBeamSpotHandle.isValid()){
    reco::BeamSpot beamSpot = *recoBeamSpotHandle;
    vertexPoint = beamSpot.position();
  }else{
    edm::LogError("UnusableBeamSpot") << "No beam spot found in Event";
  }
  }

  if(fUseVertex)
  {
  // get the Vertex
  edm::Handle<edm::View<reco::Vertex> > recoVertexHandle;
  iEvent.getByLabel(vertexLabel,recoVertexHandle);
  if ((recoVertexHandle.isValid()) && (recoVertexHandle->size()>0)){
    reco::Vertex vertex = recoVertexHandle->at(0);
    vertexPoint = vertex.position();
  }else{
    edm::LogError("UnusableVertex") << "No vertex found in Event";
  }
  }

  if(fIsRecoCandidate)
  {
    edm::Handle<edm::View<reco::RecoCandidate> > candidateHandle;
    iEvent.getByLabel(trackLabel, candidateHandle);
    if ((candidateHandle.isValid())&&(candidateHandle->size()>0)){
      double maxpt=0.;
      unsigned i_maxpt=0;
      for (unsigned i = 0; i < candidateHandle->size(); ++i) {
        double pt=candidateHandle->ptrAt(i)->pt();
        if(pt>maxpt)
	{
	  i_maxpt=i;
	  maxpt=pt;
	}
      }
      const reco::Track* track = dynamic_cast<const reco::Track*>(candidateHandle->ptrAt(i_maxpt)->bestTrack());
      vertexPoint.SetZ(vertexPoint.z()+track->dz(vertexPoint));
      math::Error<3>::type noErrors;
      reco::Vertex v(vertexPoint, noErrors);
      vColl.push_back(v);
    }
  } else {
    edm::Handle<edm::View<reco::Track> > trackHandle;
    iEvent.getByLabel(trackLabel, trackHandle);
    if ((trackHandle.isValid())&&(trackHandle->size()>0)){
      double maxpt=0.;
      unsigned i_maxpt=0;
      for (unsigned i = 0; i < trackHandle->size(); ++i) {
        double pt=trackHandle->ptrAt(i)->pt();
        if(pt>maxpt)
	{
	  i_maxpt=i;
	  maxpt=pt;
	}
      }
      const reco::Track* track = dynamic_cast<const reco::Track*>(&*trackHandle->ptrAt(i_maxpt));
      vertexPoint.SetZ(vertexPoint.z()+track->dz(vertexPoint));
      math::Error<3>::type noErrors;
      reco::Vertex v(vertexPoint, noErrors);
      vColl.push_back(v);
    }
  }

  // provide beamspot or primary vertex if no candidate found
  //if(vColl.size()==0)
  //{
  //    math::Error<3>::type noErrors;
  //    reco::Vertex v(vertexPoint, noErrors);
  //    vColl.push_back(v);
  //}

  if(fVerbose){
    int ivtx=0;
    for(reco::VertexCollection::const_iterator v=vColl.begin(); 
	v!=vColl.end(); ++v){
      std::cout << "recvtx "<< ivtx++ 
		<< " x "  << std::setw(6) << v->position().x() 
		<< " dx " << std::setw(6) << v->xError()
		<< " y "  << std::setw(6) << v->position().y() 
		<< " dy " << std::setw(6) << v->yError()
		<< " z "  << std::setw(6) << v->position().z() 
		<< " dz " << std::setw(6) << v->zError()
		<< std::endl;
    }
  }

  
  *result = vColl;
  iEvent.put(result);
  
}
