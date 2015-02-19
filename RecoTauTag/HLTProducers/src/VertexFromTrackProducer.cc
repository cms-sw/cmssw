#include "RecoTauTag/HLTProducers/interface/VertexFromTrackProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

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
VertexFromTrackProducer::VertexFromTrackProducer(const edm::ParameterSet& conf) : 
  trackToken( consumes<edm::View<reco::Track> >(conf.getParameter<edm::InputTag>("trackLabel")) ),
  candidateToken( consumes<edm::View<reco::RecoCandidate> >(conf.getParameter<edm::InputTag>("trackLabel")) ),
  triggerFilterElectronsSrc( consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("triggerFilterElectronsSrc")) ),
  triggerFilterMuonsSrc( consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("triggerFilterMuonsSrc")) ),
  vertexLabel( consumes<edm::View<reco::Vertex> >(conf.getParameter<edm::InputTag>("vertexLabel")) ),
  beamSpotLabel( consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotLabel")) ),
  fIsRecoCandidate( conf.getParameter<bool>("isRecoCandidate") ),
  fUseBeamSpot( conf.getParameter<bool>("useBeamSpot") ),
  fUseVertex( conf.getParameter<bool>("useVertex") ),
  fUseTriggerFilterElectrons( conf.getParameter<bool>("useTriggerFilterElectrons") ),
  fUseTriggerFilterMuons( conf.getParameter<bool>("useTriggerFilterMuons") ),
  fVerbose( conf.getUntrackedParameter<bool>("verbose", false) )
{
  edm::LogInfo("PVDebugInfo") 
    << "Initializing  VertexFromTrackProducer" << "\n";

  produces<reco::VertexCollection>();

}


VertexFromTrackProducer::~VertexFromTrackProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
VertexFromTrackProducer::produce(edm::StreamID iStreamId, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace edm;

  std::auto_ptr<reco::VertexCollection> result(new reco::VertexCollection);
  reco::VertexCollection vColl;

  math::XYZPoint vertexPoint;
  bool vertexAvailable = false;

  // get the BeamSpot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotLabel,recoBeamSpotHandle);
  if (recoBeamSpotHandle.isValid()){
    reco::BeamSpot beamSpot = *recoBeamSpotHandle;
    vertexPoint = beamSpot.position();
  }else{
    edm::LogError("UnusableBeamSpot") << "No beam spot found in Event";
  }

  if(fUseVertex)
  {
    // get the Vertex
    edm::Handle<edm::View<reco::Vertex> > recoVertexHandle;
    iEvent.getByToken(vertexLabel,recoVertexHandle);
    if ((recoVertexHandle.isValid()) && (recoVertexHandle->size()>0)){
      reco::Vertex vertex = recoVertexHandle->at(0);
      vertexPoint = vertex.position();
      vertexAvailable = true;
    }else {
      edm::LogInfo("UnusableVertex")
	<< "No vertex found in Event, beam spot used instaed" << "\n";
    }
  }

  const reco::Track* track = 0;
  if(fIsRecoCandidate)
  {
    edm::Handle<edm::View<reco::RecoCandidate> > candidateHandle;
    iEvent.getByToken(candidateToken, candidateHandle);
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
      track = dynamic_cast<const reco::Track*>(candidateHandle->ptrAt(i_maxpt)->bestTrack());
    }
  } 
  else if(fUseTriggerFilterElectrons) {
    edm::Handle<trigger::TriggerFilterObjectWithRefs> triggerfilter;
    iEvent.getByToken(triggerFilterElectronsSrc, triggerfilter);
    std::vector<reco::ElectronRef> recocandidates;
    triggerfilter->getObjects(trigger::TriggerElectron,recocandidates);
    if ((recocandidates.size()>0)){
      double maxpt=0.;
      unsigned i_maxpt=0;
      for (unsigned i = 0; i < recocandidates.size(); ++i) {
	double pt=recocandidates.at(i)->pt();
	if(pt>maxpt) 
	  {
	    i_maxpt=i;
	    maxpt=pt;
	  }
	track = dynamic_cast<const reco::Track*>(recocandidates.at(i_maxpt)->bestTrack());
      }
    }
  }  
  else if(fUseTriggerFilterMuons) {
    edm::Handle<trigger::TriggerFilterObjectWithRefs> triggerfilter;
    iEvent.getByToken(triggerFilterMuonsSrc, triggerfilter);
    std::vector<reco::RecoChargedCandidateRef> recocandidates;
    triggerfilter->getObjects(trigger::TriggerMuon,recocandidates);
    if ((recocandidates.size()>0)){
      double maxpt=0.;
      unsigned i_maxpt=0;
      for (unsigned i = 0; i < recocandidates.size(); ++i) {
	double pt=recocandidates.at(i)->pt();
	if(pt>maxpt) 
	  {
	    i_maxpt=i;
	    maxpt=pt;
	  }
	track = dynamic_cast<const reco::Track*>(recocandidates.at(i_maxpt)->bestTrack());
      }
    }
  }
  else {
    edm::Handle<edm::View<reco::Track> > trackHandle;
    iEvent.getByToken(trackToken, trackHandle);
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
      track = dynamic_cast<const reco::Track*>(&*trackHandle->ptrAt(i_maxpt));
    }
  }

  if(track) {
    if(fUseBeamSpot || (fUseVertex && vertexAvailable) ) {
      vertexPoint.SetZ(vertexPoint.z()+track->dz(vertexPoint));
    }
    else {
      vertexPoint.SetZ(track->vz());
    }
  }
  math::Error<3>::type noErrors;
  reco::Vertex v(vertexPoint, noErrors);
  vColl.push_back(v);

  // provide beamspot or primary vertex if no candidate found
  //if(vColl.size()==0)
  //{
  //    math::Error<3>::type noErrors;
  //    reco::Vertex v(vertexPoint, noErrors);
  //    vColl.push_back(v);
  //}

  if(fVerbose){
    int ivtx=0;
    edm::LogInfo("PVDebugInfo")<< "Vertices by VertexFromTrackProducer: \n"; 
    for(reco::VertexCollection::const_iterator v=vColl.begin(); 
	v!=vColl.end(); ++v){
      edm::LogInfo("PVDebugInfo")<< "\t" 
				 << "recvtx "<< ivtx++ 
				 << " x "  << std::setw(6) << v->position().x() 
				 << " dx " << std::setw(6) << v->xError()
				 << " y "  << std::setw(6) << v->position().y() 
				 << " dy " << std::setw(6) << v->yError()
				 << " z "  << std::setw(6) << v->position().z() 
				 << " dz " << std::setw(6) << v->zError()
				 << " \n ";
    }
  }

  
  *result = vColl;
  iEvent.put(result);
  
}

void
VertexFromTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{

  edm::ParameterSetDescription desc;

  desc.add<bool>("isRecoCandidate",false)->setComment("If isRecoCandidate=True \"trackLabel\" is used and assumed to be collection of candidates.\nOtherwise it is assumed that \"trackLabel\" is collection of tracks and is used when useTriggerFilterElectrons=False and useTriggerFilterMuons=False");
  desc.add<edm::InputTag>("trackLabel",edm::InputTag("hltL3MuonCandidates"))->setComment("Collection of tracks or candidates");
  desc.add<bool>("useTriggerFilterElectrons",false)->setComment("Use leading electron passing \"triggerFilterElectronsSrc\" filter to determine z vertex position");
  desc.add<edm::InputTag>("triggerFilterElectronsSrc",edm::InputTag("hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoL1JetTrackIsoFilter"))->setComment("Name of electron filter");
  desc.add<bool>("useTriggerFilterMuons",true)->setComment("Use leading muon passing \"triggerFilterMuonsSrc\" filter to determine z vertex position");
  desc.add<edm::InputTag>("triggerFilterMuonsSrc",edm::InputTag("hltSingleMuIsoL3IsoFiltered15"))->setComment("Name of muon filter");
  desc.add<bool>("useBeamSpot",true)->setComment("Use beam spot for x/y vertex position");
  desc.add<edm::InputTag>("beamSpotLabel",edm::InputTag("hltOnlineBeamSpot"))->setComment("Beamspot collection");
  desc.add<bool>("useVertex",true)->setComment("Use vertex for x/y vertex position (beam spot is used when PV does not exit)");
  desc.add<edm::InputTag>("vertexLabel",edm::InputTag("hltPixelVertices"))->setComment("Vertex collection");

  desc.addUntracked<bool>("verbose",false)->setComment("Switch on/off verbosity");
  descriptions.setComment("This module produces vertex with z-coordinate determined with the highest-Pt lepton track and x/y-coordinates taken from BeamSpot/Vertex");
  descriptions.add("hltVertexFromTrackProducer",desc);

}
