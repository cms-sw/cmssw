#ifndef MuonIsolation_TrackExtractor_H
#define MuonIsolation_TrackExtractor_H

#include <string>
#include <vector>


#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"

namespace muonisolation {

class TrackExtractor : public MuIsoExtractor {

public:

  TrackExtractor() { }
  TrackExtractor( double aDdiff_r, double aDiff_z, double aDR_match, double aDR_Veto,
      std::string aTrackCollectionLabel, std::string aDepisitLabel);

  virtual ~TrackExtractor(){}

  virtual std::vector<reco::MuIsoDeposit> deposits( const edm::Event & ev, const reco::Track & track, 
      const std::vector<muonisolation::Direction> & vetoDirs, double coneSize) const; 

private:
  
  void fillDeposits( reco::MuIsoDeposit & deposit, const reco::TrackCollection & tracks,
     const std::vector<muonisolation::Direction> & vetos) const;

private:
  double theDiff_r, theDiff_z, theDR_Match, theDR_Veto;
  std::string theTrackCollectionLabel; // Isolation track Collection Label
  std::string theDepositLabel;         // name for deposit
};

}

#endif
