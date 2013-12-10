#ifndef MuonIsolation_TrackExtractor_H
#define MuonIsolation_TrackExtractor_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

namespace muonisolation {

class TrackExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  TrackExtractor(){};
  TrackExtractor(const edm::ParameterSet& par,edm::ConsumesCollector& iC);

  virtual ~TrackExtractor(){}

  virtual void fillVetos (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::TrackCollection & track) {}

  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & track)const;

  virtual reco::IsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & muon) const;


private:
  reco::IsoDeposit::Veto veto( const reco::IsoDeposit::Direction & dir) const;
private:
  // Parameter set
  edm::InputTag  theTrackCollectionTag; //! Track Collection Label
  std::string theDepositLabel;         //! name for deposit
  double theDiff_r;                    //! transverse distance to vertex
  double theDiff_z;                    //! z distance to vertex
  double theDR_Max;                    //! Maximum cone angle for deposits
  double theDR_Veto;                   //! Veto cone angle
  std::string theBeamlineOption;       //! "NONE", "BeamSpotFromEvent"
  edm::InputTag theBeamSpotLabel;      //! BeamSpot name
  unsigned int theNHits_Min;                   //! trk.numberOfValidHits >= theNHits_Min
  double theChi2Ndof_Max;              //! trk.normalizedChi2 < theChi2Ndof_Max
  double theChi2Prob_Min;              //! ChiSquaredProbability(trk.chi2,trk.ndof) > theChi2Prob_Min
  double thePt_Min;                    //! min track pt to include into iso deposit
  edm::EDGetTokenT<edm::View<reco::Track> > trackToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;


};

}

#endif
