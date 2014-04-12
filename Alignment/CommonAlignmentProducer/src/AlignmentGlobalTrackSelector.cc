//Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"

//DataFormats
#include <DataFormats/Candidate/interface/Particle.h>
#include <DataFormats/Candidate/interface/Candidate.h>
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/JetReco/interface/CaloJet.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <DataFormats/RecoCandidate/interface/RecoCandidate.h> //for the get<TrackRef>() Call

#include <DataFormats/Math/interface/deltaR.h>

//STL
#include <math.h>

#include "Alignment/CommonAlignmentProducer/interface/AlignmentGlobalTrackSelector.h"


using namespace std;
using namespace edm;   

// constructor ----------------------------------------------------------------
AlignmentGlobalTrackSelector::AlignmentGlobalTrackSelector(const edm::ParameterSet & cfg) :
  theGMFilterSwitch(cfg.getParameter<bool>("applyGlobalMuonFilter")),
  theIsoFilterSwitch(cfg.getParameter<bool>("applyIsolationtest")),
  theJetCountFilterSwitch(cfg.getParameter<bool>("applyJetCountFilter")),
  theMuonSource("muons"),
  theJetIsoSource("fastjet6CaloJets"),
  theJetCountSource("fastjet6CaloJets")
{
  if (theGMFilterSwitch || theIsoFilterSwitch || theJetCountFilterSwitch)
    LogDebug("Alignment") << "> applying global Trackfilter ...";
 
  if (theGMFilterSwitch) {
    theMuonSource = cfg.getParameter<InputTag>("muonSource");
    theMaxTrackDeltaR =cfg.getParameter<double>("maxTrackDeltaR");
    theMinGlobalMuonCount = cfg.getParameter<int>("minGlobalMuonCount");
    LogDebug("Alignment") << ">  GlobalMuonFilter : source, maxTrackDeltaR, min. Count       : "
			  << theMuonSource << " , "
			  << theMaxTrackDeltaR << " , "
			  << theMinIsolatedCount;
  }
  
  if (theIsoFilterSwitch) {
    theJetIsoSource = cfg.getParameter<InputTag>("jetIsoSource");
    theMaxJetPt = cfg.getParameter<double>("maxJetPt");
    theMinJetDeltaR = cfg.getParameter<double>("minJetDeltaR");
    theMinIsolatedCount = cfg.getParameter<int>("minIsolatedCount");
    LogDebug("Alignment") << ">  Isolationtest    : source, maxJetPt, minJetDeltaR, min. Count: "
			  << theJetIsoSource << " , "
			  << theMaxJetPt << " ,"
			  << theMinJetDeltaR << " ,"
			  << theMinGlobalMuonCount;
  }
  
  if (theJetCountFilterSwitch) {
    theJetCountSource = cfg.getParameter<InputTag>("jetCountSource");
    theMinJetPt = cfg.getParameter<double>("minJetPt");
    theMaxJetCount = cfg.getParameter<int>("maxJetCount");
    LogDebug("Alignment") << ">  JetCountFilter   : source, minJetPt, maxJetCount             : "
			  << theJetCountSource << " , "
			  << theMinJetPt << " ,"
			  << theMaxJetCount;  
  }
}

// destructor -----------------------------------------------------------------
AlignmentGlobalTrackSelector::~AlignmentGlobalTrackSelector()
{}

///returns if any of the Filters is used.
bool AlignmentGlobalTrackSelector::useThisFilter()
{
  return theGMFilterSwitch || theIsoFilterSwitch|| theJetCountFilterSwitch;
}

// do selection ---------------------------------------------------------------
AlignmentGlobalTrackSelector::Tracks 
AlignmentGlobalTrackSelector::select(const Tracks& tracks, const edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  Tracks result = tracks;

  if (theGMFilterSwitch) result = findMuons(result, iEvent);
  if (theIsoFilterSwitch) result = checkIsolation(result, iEvent);
  if (theJetCountFilterSwitch) result = checkJetCount(result, iEvent);
  LogDebug("Alignment") << ">  Global: tracks all, kept: " << tracks.size() << ", " << result.size();
//  LogDebug("Alignment")<<">  o kept:";
//  printTracks(result);
  
  return result;
}

///filter for Tracks that match the Track of a global Muon
AlignmentGlobalTrackSelector::Tracks 
AlignmentGlobalTrackSelector::findMuons(const Tracks& tracks, const edm::Event& iEvent) const
{
  Tracks result;
  Tracks globalMuons;

  //fill globalMuons with muons
  Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(theMuonSource, muons);

  if (muons.isValid()) {
    for (reco::MuonCollection::const_iterator itMuon = muons->begin(); 
	 itMuon != muons->end();
	 ++itMuon) {
      const reco::Track* muonTrack = (*itMuon).get<reco::TrackRef>().get();
      if (!muonTrack) {
	LogDebug("Alignment") << "@SUB=AlignmentGlobalTrackSelector::findMuons"
			      << "Found muon without track: Standalone Muon!";
      } else {
	globalMuons.push_back(muonTrack);
      }
    }
  } else {
    LogError("Alignment") << "@SUB=AlignmentGlobalTrackSelector::findMuons"
			  <<">  could not optain mounCollection!";
  }

  result = this->matchTracks(tracks, globalMuons);
  
  if (static_cast<int>(result.size()) < theMinGlobalMuonCount) result.clear();

  return result;
}

///returns only isolated tracks in [cands]
AlignmentGlobalTrackSelector::Tracks 
AlignmentGlobalTrackSelector::checkIsolation(const Tracks& cands,const edm::Event& iEvent) const
{
  Tracks result; result.clear();

  Handle<reco::CaloJetCollection> jets;
  iEvent.getByLabel(theJetIsoSource, jets);

  if (jets.isValid()) {
    for (Tracks::const_iterator it = cands.begin();
	 it!=cands.end();
	 ++it) {
      bool isolated = true;
      for (reco::CaloJetCollection::const_iterator itJet = jets->begin();
	   itJet!=jets->end();
	   ++itJet) 
	isolated &= !((*itJet).pt() > theMaxJetPt && deltaR(*(*it),(*itJet)) < theMinJetDeltaR);
      
      if (isolated)
	result.push_back(*it);
    }
    //    LogDebug("Alignment") << "D  Found "<<result.size()<<" isolated of "<< cands.size()<<" Tracks!";   
    
  } else
    LogError("Alignment")<< "@SUB=AlignmentGlobalTrackSelector::checkIsolation"
			 << "> could not optain jetCollection!";

  if (static_cast<int>(result.size()) < theMinIsolatedCount) result.clear();

  return result;
}

///returns [tracks] if there are less than theMaxCount Jets with theMinJetPt and an empty set if not
AlignmentGlobalTrackSelector::Tracks 
AlignmentGlobalTrackSelector::checkJetCount(const Tracks& tracks, const edm::Event& iEvent) const
{
  Tracks result; result.clear();

  Handle<reco::CaloJetCollection> jets;
  iEvent.getByLabel(theJetCountSource, jets);

  if (jets.isValid()) {
    int jetCount = 0;
    for (reco::CaloJetCollection::const_iterator itJet = jets->begin();
	 itJet!=jets->end();
	 ++itJet) {
      if ((*itJet).pt() > theMinJetPt)
	jetCount++;
    }

    if (jetCount <= theMaxJetCount)
      result = tracks;

    LogDebug("Alignment") << "> found " << jetCount << " Jets";
  } else
    LogError("Alignment") << "@SUB=AlignmentGlobalTrackSelector::checkJetCount"
			  << ">  could not optain jetCollection!";

  return result;
}

//===================HELPERS===================

///matches [src] with [comp] returns collection with matching Tracks coming from [src]
AlignmentGlobalTrackSelector::Tracks
AlignmentGlobalTrackSelector::matchTracks(const Tracks& src, const Tracks& comp) const
{
  Tracks result;
  for (Tracks::const_iterator itComp = comp.begin();
       itComp!=comp.end();
       ++itComp) {
    int match = -1;
    double min = theMaxTrackDeltaR;
    for (unsigned int i=0;i<src.size();i++) {
      // LogDebug("Alignment") << "> Trackmatch dist: "<<deltaR(src.at(i),*itComp);
      if(min > deltaR(*(src.at(i)),*(*itComp))){
	min = deltaR(*(src.at(i)),*(*itComp));
	match = static_cast<int>(i);
      }
    }
    if (match > -1)
      result.push_back(src.at(match)); 
  }
  return result;
}

///print Information on Track-Collection
void AlignmentGlobalTrackSelector::printTracks(const Tracks& col) const
{
  int count = 0;
  LogDebug("Alignment") << ">......................................";
  for (Tracks::const_iterator it = col.begin();
       it < col.end();
       ++it,++count) {
    LogDebug("Alignment") 
      << ">  Track No. " << count << ": p = (" 
      << (*it)->px() << "," 
      << (*it)->py() << ","
      << (*it)->pz() << ")\n"
      << ">                        pT = "
      << (*it)->pt() << " eta = "
      << (*it)->eta() << " charge = "
      << (*it)->charge();    
  }
  LogDebug("Alignment") << ">......................................";
}
