/**  \class L3MuonCandidateProducer
 * 
 *   Intermediate step in the L3 muons selection.
 *   This class takes the L3 muons (which are reco::Tracks) 
 *   and creates the correspondent reco::RecoChargedCandidate.
 *
 *   Riccardo's comment:
 *   The separation between the L3MuonProducer and this class allows
 *   the interchangeability of the L3MuonProducer and the StandAloneMuonProducer
 *   This class is supposed to be removed once the
 *   L3/STA comparison will be done, then the RecoChargedCandidate
 *   production will be put into the L3MuonProducer class.
 *
 *
 *   \author  J.Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonCandidateProducer.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

// Input and output collections
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <string>
#include <algorithm>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L3MuonCandidateProducer::L3MuonCandidateProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|L3MuonCandidateProducer")<<" constructor called";

  // StandAlone Collection Label
  theL3CollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  theL3LinksLabel = parameterSet.existsAs<InputTag>("InputLinksObjects") ? parameterSet.getParameter<InputTag>("InputLinksObjects") : InputTag( "unused");
  const std::string & muon_track_for_momentum = parameterSet.existsAs<std::string>("MuonPtOption") ?  parameterSet.getParameter<std::string>("MuonPtOption") : "Global";
  if (muon_track_for_momentum == std::string("Tracker"))
    type=InnerTrack;
  else if (muon_track_for_momentum == std::string("Standalone"))
    type=OuterTrack;
  else if (muon_track_for_momentum == std::string("Global")) 
    type=CombinedTrack;
  else {
    LogError("Muon|RecoMuon|L3MuonCandidateProducer")<<"invalid value for MuonPtOption, please choose among 'Tracker', 'Standalone', 'Global'";
    type=CombinedTrack;
  }

  produces<RecoChargedCandidateCollection>();
}
  
/// destructor
L3MuonCandidateProducer::~L3MuonCandidateProducer(){
  LogTrace("Muon|RecoMuon|L3MuonCandidateProducer")<<" L3MuonCandidateProducer destructor called";
}


/// reconstruct muons
void L3MuonCandidateProducer::produce(Event& event, const EventSetup& eventSetup){
  const string metname = "Muon|RecoMuon|L3MuonCandidateProducer";
  
  // Take the L3 container
  LogTrace(metname)<<" Taking the L3/GLB muons: "<<theL3CollectionLabel.label();
  Handle<TrackCollection> tracks; 
  event.getByLabel(theL3CollectionLabel,tracks);

  edm::Handle<reco::MuonTrackLinksCollection> links; 
  bool useLinks = event.getByLabel(theL3LinksLabel,links);

  // Create a RecoChargedCandidate collection
  LogTrace(metname)<<" Creating the RecoChargedCandidate collection";
  auto_ptr<RecoChargedCandidateCollection> candidates( new RecoChargedCandidateCollection()); 
  LogDebug(metname) << " size = " << tracks->size() ;  
  for (unsigned int i=0; i<tracks->size(); i++) {
    TrackRef inRef(tracks,i);
    TrackRef tkRef = TrackRef();

    if (useLinks) {
      for(reco::MuonTrackLinksCollection::const_iterator link = links->begin();
	  link != links->end(); ++link){ LogDebug(metname) << " i = " << i ;  

	if(!link->trackerTrack().isNull()) LogTrace(metname) << " link tk pt " << link->trackerTrack()->pt();
	if(!link->standAloneTrack().isNull()) LogTrace(metname) << " sta pt " << link->standAloneTrack()->pt();
	if(!link->globalTrack().isNull()) LogTrace(metname) << " global pt " << link->globalTrack()->pt();
	if(!inRef.isNull()) LogTrace(metname) << " inRef pt " << inRef->pt() ;

	double dR = (!link->globalTrack().isNull()) ? deltaR(inRef->eta(),inRef->phi(),link->globalTrack()->eta(),link->globalTrack()->phi()) : 999.;
	if(tracks->size() > 1 && !link->globalTrack().isNull() && dR < 0.02 && abs(inRef->pt() - link->globalTrack()->pt())/inRef->pt() < 0.001) {
	  LogTrace(metname) << " *** pt matches *** ";
	  switch(type) {
	  case InnerTrack:    tkRef = link->trackerTrack();    break;
	  case OuterTrack:    tkRef = link->standAloneTrack(); break;
	  case CombinedTrack: tkRef = link->globalTrack();     break;
	  default:            tkRef = link->globalTrack();     break;
	  }
	}
      }
    }
    if(tkRef.isNull()) {
      LogDebug(metname) << "tkRef is NULL";
      tkRef = inRef;
    }
    LogDebug(metname) << "tkRef Used For Momentum pt " << tkRef->pt() << " inRef from the input collection pt " << inRef->pt(); 

    Particle::Charge q = tkRef->charge();
    Particle::LorentzVector p4(tkRef->px(), tkRef->py(), tkRef->pz(), tkRef->p());
    Particle::Point vtx(tkRef->vx(),tkRef->vy(), tkRef->vz());
    
    int pid = 13;
    if(abs(q)==1) pid = q < 0 ? 13 : -13;
    else LogWarning(metname) << "L3MuonCandidate has charge = "<<q;
    RecoChargedCandidate cand(q, p4, vtx, pid); 

    //set the inRef as the RecoChargedCandidate ref so that the isolation maps 
    //work in downstream filters
    cand.setTrack(inRef);
    candidates->push_back(cand);
  }
  
  event.put(candidates);
 
  LogTrace(metname)<<" Event loaded"
		   <<"================================";
}

