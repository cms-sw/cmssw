#ifndef EgammaIsolationProducers_EgammaTrackExtractor_H
#define EgammaIsolationProducers_EgammaTrackExtractor_H

#include <string>
#include <vector>


#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace egammaisolation {

class EgammaTrackExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  EgammaTrackExtractor(){};
  EgammaTrackExtractor(const edm::ParameterSet& par, edm::ConsumesCollector && iC) :
    EgammaTrackExtractor(par, iC) {}
  EgammaTrackExtractor(const edm::ParameterSet& par, edm::ConsumesCollector & iC);

  virtual ~EgammaTrackExtractor(){}

  virtual void fillVetos (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::TrackCollection & track) {}

  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & track)const;

  virtual reco::IsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & muon) const {
        edm::LogWarning("EgammaIsolationAlgos|EgammaTrackExtractor")
           << "This Function is not implemented, bad IsoDeposit Returned";
        return reco::IsoDeposit( reco::isodeposit::Direction(1,1) );
      }

  virtual reco::IsoDeposit deposit (const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & muon) const;

private:
  reco::IsoDeposit::Veto veto( const reco::IsoDeposit::Direction & dir) const;
private:
  // Parameter set
  edm::EDGetTokenT<edm::View<reco::Track> > theTrackCollectionToken;      //! Track Collection Label
  std::string theDepositLabel;              //! name for deposit
  double minCandEt_;                         //! minimum candidate et
  double theDiff_r;                         //! transverse distance to vertex
  double theDiff_z;                         //! z distance to vertex
  double theDR_Max;                         //! Maximum cone angle for deposits
  double theDR_Veto;                        //! Veto cone angle
  std::string theBeamlineOption;            //! "NONE", "BeamSpotFromEvent"
  edm::InputTag barrelEcalHitsTag_;
  edm::InputTag endcapEcalHitsTag_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;           //! BeamSpot name
  unsigned int theNHits_Min;                        //! trk.numberOfValidHits >= theNHits_Min
  double theChi2Ndof_Max;                   //! trk.normalizedChi2 < theChi2Ndof_Max
  double theChi2Prob_Min;                   //! ChiSquaredProbability(trk.chi2,trk.ndof) > theChi2Prob_Min
  double thePt_Min;                         //! min track pt to include into iso deposit
  std::vector<double> paramForIsolBarrel_;   //! Barrel requirements to determine if isolated for selective filling
  std::vector<double> paramForIsolEndcap_;   //! Endcap requirements to determine if isolated for selective filling
  std::string dzOptionString;
  int dzOption;
};

}

#endif
