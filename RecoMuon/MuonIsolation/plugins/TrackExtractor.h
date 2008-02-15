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

  TrackExtractor(){};
  TrackExtractor(const edm::ParameterSet& par);

  virtual ~TrackExtractor(){}

  virtual void fillVetos (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::TrackCollection & track) {}

  virtual reco::MuIsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & track)const;

  virtual reco::MuIsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & muon) const;

private:
  reco::MuIsoDeposit::Veto veto( const reco::MuIsoDeposit::Direction & dir) const;
private:
  // Parameter set
  edm::InputTag theTrackCollectionTag; //! Track Collection Label
  std::string theDepositLabel;         //! name for deposit
  double theDiff_r;                    //! transverse distance to vertex
  double theDiff_z;                    //! z distance to vertex
  double theDR_Max;                    //! Maximum cone angle for deposits
  double theDR_Veto;                   //! Veto cone angle
  std::string theBeamlineOption;       //! "NONE", "BeamSpotFromEvent"
  edm::InputTag theBeamSpotLabel;      //! BeamSpot name
  uint theNHits_Min;                   //! trk.numberOfValidHits >= theNHits_Min
  double theChi2Ndof_Max;              //! trk.normalizedChi2 < theChi2Ndof_Max
  double theChi2Prob_Min;              //! ChiSquaredProbability(trk.chi2,trk.ndof) > theChi2Prob_Min
  double thePt_Min;                    //! min track pt to include into iso deposit
};

}

#endif
