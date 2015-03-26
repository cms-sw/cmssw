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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonCandidateProducer.h"


#include "DataFormats/Math/interface/deltaR.h"

// Input and output collections
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <string>
#include <algorithm>

using namespace edm;
using namespace std;
using namespace reco;

static const char category[] = "Muon|RecoMuon|L3MuonCandidateProducer";

/// constructor with config
L3MuonCandidateProducer::L3MuonCandidateProducer(const ParameterSet& parameterSet){
  LogTrace(category)<<" constructor called";

  // StandAlone Collection Label
  theL3CollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  trackToken_ = consumes<reco::TrackCollection>(theL3CollectionLabel);
 
  // use links
  theUseLinks = parameterSet.existsAs<InputTag>("InputLinksObjects");
  if (theUseLinks) {
    theL3LinksLabel = parameterSet.getParameter<InputTag>("InputLinksObjects");
    linkToken_ = consumes<reco::MuonTrackLinksCollection>(theL3LinksLabel);
    if (theL3LinksLabel.label() == "" or theL3LinksLabel.label() == "unused")
      theUseLinks = false;
  }

  // use global, standalone or tracker pT/4-vector assignment
  const std::string & muon_track_for_momentum = parameterSet.existsAs<std::string>("MuonPtOption") ?  parameterSet.getParameter<std::string>("MuonPtOption") : "Global";
  if (muon_track_for_momentum == std::string("Tracker"))
    theType=InnerTrack;
  else if (muon_track_for_momentum == std::string("Standalone"))
    theType=OuterTrack;
  else if (muon_track_for_momentum == std::string("Global"))
    theType=CombinedTrack;
  else {
    LogError(category)<<"invalid value for MuonPtOption, please choose among 'Tracker', 'Standalone', 'Global'";
    theType=CombinedTrack;
  }

  produces<RecoChargedCandidateCollection>();
}

/// destructor
L3MuonCandidateProducer::~L3MuonCandidateProducer(){
  LogTrace(category)<<" L3MuonCandidateProducer destructor called";
}


/// reconstruct muons
void L3MuonCandidateProducer::produce(StreamID, Event& event, const EventSetup& eventSetup) const{
  // Take the L3 container
  LogTrace(category)<<" Taking the L3/GLB muons: "<<theL3CollectionLabel.label();
  Handle<TrackCollection> tracks;
  event.getByToken(trackToken_,tracks);

  edm::Handle<reco::MuonTrackLinksCollection> links;
  if (theUseLinks)
    event.getByToken(linkToken_, links);

  // Create a RecoChargedCandidate collection
  LogTrace(category)<<" Creating the RecoChargedCandidate collection";
  auto_ptr<RecoChargedCandidateCollection> candidates( new RecoChargedCandidateCollection());
  LogDebug(category) << " size = " << tracks->size();
  for (unsigned int i=0; i<tracks->size(); i++) {
    TrackRef inRef(tracks,i);
    TrackRef tkRef = TrackRef();

    if (theUseLinks) {
      for(reco::MuonTrackLinksCollection::const_iterator link = links->begin();
          link != links->end(); ++link){ LogDebug(category) << " i = " << i;

        if (not link->trackerTrack().isNull()) LogTrace(category) << " link tk pt " << link->trackerTrack()->pt();
        if (not link->standAloneTrack().isNull()) LogTrace(category) << " sta pt " << link->standAloneTrack()->pt();
        if (not link->globalTrack().isNull()) LogTrace(category) << " global pt " << link->globalTrack()->pt();
        if (not inRef.isNull()) LogTrace(category) << " inRef pt " << inRef->pt();

        if (link->globalTrack().isNull()) {
          edm::LogError(category) << "null reference to the global track";
          // skip this candidate
          continue;
        }

        float dR  = deltaR(inRef->eta(),inRef->phi(),link->globalTrack()->eta(),link->globalTrack()->phi());
        float dPt = abs(inRef->pt() - link->globalTrack()->pt())/inRef->pt();
        if (dR < 0.02 and dPt < 0.001) {
          LogTrace(category) << " *** pt matches *** ";
          switch(theType) {
            case InnerTrack:    tkRef = link->trackerTrack();    break;
            case OuterTrack:    tkRef = link->standAloneTrack(); break;
            case CombinedTrack: tkRef = link->globalTrack();     break;
            default:            tkRef = link->globalTrack();     break;
          }
        }
      }
      if (tkRef.isNull()) {
        edm::LogWarning(category) << "null reference to the linked track, reverting to old behaviour";
        tkRef = inRef;
      }
    } else {
      // theUseLinks is false
      tkRef = inRef;
    }
    LogDebug(category) << "tkRef Used For Momentum pt " << tkRef->pt() << " inRef from the input collection pt " << inRef->pt();

    Particle::Charge q = tkRef->charge();
    Particle::LorentzVector p4(tkRef->px(), tkRef->py(), tkRef->pz(), tkRef->p());
    Particle::Point vtx(tkRef->vx(),tkRef->vy(), tkRef->vz());

    int pid = 13;
    if(abs(q)==1) pid = q < 0 ? 13 : -13;
    else LogWarning(category) << "L3MuonCandidate has charge = "<<q;
    RecoChargedCandidate cand(q, p4, vtx, pid);

    //set the inRef as the RecoChargedCandidate ref so that the isolation maps
    //work in downstream filters
    cand.setTrack(inRef);
    candidates->push_back(cand);
  }

  event.put(candidates);

  LogTrace(category)<<" Event loaded"
                   <<"================================";
}

