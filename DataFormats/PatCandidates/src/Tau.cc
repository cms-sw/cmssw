//
// $Id: Tau.cc,v 1.5.2.1 2008/06/03 20:08:24 gpetrucc Exp $
//

#include "DataFormats/PatCandidates/interface/Tau.h"


using namespace pat;


/// default constructor
Tau::Tau() :
    Lepton<TauType>(),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false),
    emEnergyFraction_(0.),
    eOverP_(0.),
    leadeOverP_(0.),
    HhotOverP_(0.),
    HtotOverP_(0.)
{
}


/// constructor from TauType
Tau::Tau(const TauType & aTau) :
    Lepton<TauType>(aTau),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false),
    emEnergyFraction_(0.),
    eOverP_(0.),
    leadeOverP_(0.),
    HhotOverP_(0.),
    HtotOverP_(0.)
{
}


/// constructor from ref to TauType
Tau::Tau(const edm::RefToBase<TauType> & aTauRef) :
    Lepton<TauType>(aTauRef),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false),
    emEnergyFraction_(0.),
    eOverP_(0.),
    leadeOverP_(0.),
    HhotOverP_(0.),
    HtotOverP_(0.)
{
}

/// constructor from ref to TauType
Tau::Tau(const edm::Ptr<TauType> & aTauRef) :
    Lepton<TauType>(aTauRef),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false),
    emEnergyFraction_(0.),
    eOverP_(0.),
    leadeOverP_(0.),
    HhotOverP_(0.),
    HtotOverP_(0.)
{
}



/// destructor
Tau::~Tau() {
}


/// override the TauType::isolationTracks method, to access the internal storage of the track
reco::TrackRefVector Tau::isolationTracks() const {
  if (embeddedIsolationTracks_) {
    reco::TrackRefVector trackRefVec;
    for (unsigned int i = 0; i < isolationTracks_.size(); i++) {
      trackRefVec.push_back(reco::TrackRef(&isolationTracks_, i));
    }
    return trackRefVec;
  } else {
    return TauType::isolationTracks();
  }
}


/// override the TauType::track method, to access the internal storage of the track
reco::TrackRef Tau::leadTrack() const {
  if (embeddedLeadTrack_) {
    return reco::TrackRef(&leadTrack_, 0);
  } else {
    return TauType::leadTrack();
  }
}


/// override the TauType::track method, to access the internal storage of the track
reco::TrackRefVector Tau::signalTracks() const {
  if (embeddedSignalTracks_) {
    reco::TrackRefVector trackRefVec;
    for (unsigned int i = 0; i < signalTracks_.size(); i++) {
      trackRefVec.push_back(reco::TrackRef(&signalTracks_, i));
    }
    return trackRefVec;
  } else {
    return TauType::signalTracks();
  }
}


/// method to store the isolation tracks internally
void Tau::embedIsolationTracks() {
  isolationTracks_.clear();
  reco::TrackRefVector trackRefVec = TauType::isolationTracks();
  for (unsigned int i = 0; i < trackRefVec.size(); i++) {
    isolationTracks_.push_back(*trackRefVec.at(i));
  }
  embeddedIsolationTracks_ = true;
}


/// method to store the isolation tracks internally
void Tau::embedLeadTrack() {
  leadTrack_.clear();
  leadTrack_.push_back(*TauType::leadTrack());
  embeddedLeadTrack_ = true;
}


/// method to store the isolation tracks internally
void Tau::embedSignalTracks(){
  signalTracks_.clear();
  reco::TrackRefVector trackRefVec = TauType::signalTracks();
  for (unsigned int i = 0; i < trackRefVec.size(); i++) {
    signalTracks_.push_back(*trackRefVec.at(i));
  }
  embeddedSignalTracks_ = true;
}
