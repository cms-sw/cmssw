#ifndef MuonIsolation_MuIsoExtractor_H
#define MuonIsolation_MuIsoExtractor_H

//
//
//


#include "FWCore/ParameterSet/interface/ParameterSet.h"
                                                                                
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

namespace muonisolation {

class MuIsoExtractor {
public:
  virtual ~MuIsoExtractor(){};
  virtual void fillVetos(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks) = 0;
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Candidate & c) const {
    reco::Track dummy(10,10,c.vertex(),c.momentum(),c.charge(), reco::Track::CovarianceMatrix());
    return deposit( ev, evSetup, dummy );
  }
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const = 0;
  virtual reco::MuIsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackRef & track) const{
    return deposit(ev, evSetup, *track);
  }
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const{
    return std::vector<reco::MuIsoDeposit>(1, deposit(ev, evSetup, track));
  }
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackRef & track) const{
    return deposits(ev, evSetup, *track);
  }
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Candidate & c) const{
    reco::Track dummy(10,10,c.vertex(),c.momentum(),c.charge(), reco::Track::CovarianceMatrix());
    return deposits( ev, evSetup, dummy );
  }
};

}
#endif
