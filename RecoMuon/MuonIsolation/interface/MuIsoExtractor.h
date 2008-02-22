#ifndef MuonIsolation_MuIsoExtractor_H
#define MuonIsolation_MuIsoExtractor_H

//
//
//


#include "FWCore/ParameterSet/interface/ParameterSet.h"
                                                                                
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

namespace muonisolation {

class MuIsoExtractor {
public:
  //! Destructor
  virtual ~MuIsoExtractor(){};


  //! fill vetoes: to exclude deposits at MuIsoDeposit creation stage
  //! check concrete extractors if it's no-op !
  virtual void fillVetos(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks) = 0;

  //! make single MuIsoDeposit based on track as input
  //! purely virtual: have to implement in concrete implementations
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, 
				     const reco::Track & track) const = 0;

  //! make single MuIsoDeposit based on trackRef as input
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, 
				     const reco::TrackBaseRef & track) const{
    return deposit(ev, evSetup, *track);
  }

  //! make single MuIsoDeposit based on a candidate as input
  //! purely virtual: have to implement in concrete implementations
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, 
				     const reco::Candidate & track) const {
    //track-based implementation as default <-- do I want this pure virtual?
    return deposit(ev, evSetup, reco::Track(10, 10, 
					    track.vertex(), track.momentum(), track.charge(),
					    reco::Track::CovarianceMatrix()));
  }

  //! make single MuIsoDeposit based on a CandidateBaseRef as input
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, 
				     const reco::CandidateBaseRef & track) const{
    return deposit(ev, evSetup, *track);
  }

  //! make multiple MuIsoDeposit(s) based on a track as input
  //! use these only if CPU-constrained
  //! for all derived types THIS METHOD HAS TO BE IMPLEMENTED at the minimum
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const{
    return std::vector<reco::MuIsoDeposit>(1, deposit(ev, evSetup, track));
  }

  //! make multiple MuIsoDeposit(s) based on a TrackBaseRef as input
  //! use these only if CPU-constrained
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackBaseRef & track) const{
    return deposits(ev, evSetup, *track);
  }

  //! make multiple MuIsoDeposit(s) based on a candidate as input
  //! use these only if CPU-constrained
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Candidate & track) const{
    return deposits(ev, evSetup, 
		    reco::Track(10, 10,  
				track.vertex(), track.momentum(), track.charge(), 
				reco::Track::CovarianceMatrix()));
  }

  //! make multiple MuIsoDeposit(s) based on a candidateBaseRef as input
  //! use these only if CPU-constrained
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::CandidateBaseRef & track) const{
    return deposits(ev, evSetup, *track);
  }

}
;

}
#endif
