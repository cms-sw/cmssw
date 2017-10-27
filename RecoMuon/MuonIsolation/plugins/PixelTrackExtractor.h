#ifndef MuonIsolation_PixelTrackExtractor_H
#define MuonIsolation_PixelTrackExtractor_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace muonisolation {

class PixelTrackExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  PixelTrackExtractor(){};
  PixelTrackExtractor(const edm::ParameterSet& par, edm::ConsumesCollector && iC);

  ~PixelTrackExtractor() override{}

  void fillVetos (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::TrackCollection & track) override {}

  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & track)const;

  reco::IsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & muon) const override;

private:
  reco::IsoDeposit::Veto veto( const reco::IsoDeposit::Direction & dir) const;

  reco::isodeposit::Direction directionAtPresetRadius(const reco::Track& tk, double bz) const;
private:
  // Parameter set
  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionToken; //! Track Collection Token
  std::string theDepositLabel;         //! name for deposit
  double theDiff_r;                    //! transverse distance to vertex
  double theDiff_z;                    //! z distance to vertex
  double theDR_Max;                    //! Maximum cone angle for deposits
  double theDR_Veto;                   //! Veto cone angle
  std::string theBeamlineOption;       //! "NONE", "BeamSpotFromEvent"
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;      //! BeamSpot name
  unsigned int theNHits_Min;                   //! trk.numberOfValidHits >= theNHits_Min
  double theChi2Ndof_Max;              //! trk.normalizedChi2 < theChi2Ndof_Max
  double theChi2Prob_Min;              //! ChiSquaredProbability(trk.chi2,trk.ndof) > theChi2Prob_Min
  double thePt_Min;                    //! min track pt to include into iso deposit

  bool thePropagateTracksToRadius;     //! If set to true will compare track eta-phi at ...
  double theReferenceRadius;           //! ... this radius

  bool theVetoLeadingTrack;             //! will veto leading track if
  double thePtVeto_Min;		        //! .. it is above this threshold
  double theDR_VetoPt;		        //!.. and is inside this cone
};

}

#endif
