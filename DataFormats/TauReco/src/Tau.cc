#include "DataFormats/TauReco/interface/Tau.h"

using namespace reco;

Tau::Tau() {
 TrackRefVector tmp;
  TrackRef leadTk;
  leadingTrack_ = leadTk;
  signalTracks_ =tmp ;
  isolationTracks_= tmp;
  selectedTracks_ = tmp;

  PFCandidateRef pfLead;
  leadingPFChargedHadron_ = pfLead;
  PFCandidateRefVector pfTmp;
  selectedSignalPFChargedHadrons_ =pfTmp;
  selectedSignalPFNeutralHadrons_=pfTmp;
  selectedSignalPFGammaCandidates_ = pfTmp;

  selectedIsolationPFChargedHadrons_ = pfTmp;
  selectedIsolationPFNeutralHadrons_ = pfTmp;
  selectedIsolationPFGammaCandidates_ = pfTmp;

  selectedPFChargedHadrons_ = pfTmp;
  selectedPFNeutralHadrons_ = pfTmp;
  selectedPFGammaCandidates_ = pfTmp;

  maximumHcalTowerEnergy_ = -1000.;
    transverseIpSignificance_leadTk_ =-1000.;
    mass_ = -1000; 
     trackerMass_ = -1000;
     sumPtIsolation_=-1000;
     emOverHadronEnergy_=-1000;
     emIsolation_=-1000;
}

Tau::Tau(Charge q, const LorentzVector & p4, const Point & vtx ) :
   RecoCandidate( q, p4, vtx, -15 * q ) {
  TrackRefVector tmp;
  TrackRef leadTk;
  leadingTrack_ = leadTk;
  signalTracks_ =tmp ;
  isolationTracks_= tmp;
  selectedTracks_ = tmp;

  PFCandidateRef pfLead;
  leadingPFChargedHadron_ = pfLead;
  PFCandidateRefVector pfTmp;
  selectedSignalPFChargedHadrons_ =pfTmp;
  selectedSignalPFNeutralHadrons_=pfTmp;
  selectedSignalPFGammaCandidates_ = pfTmp;

  selectedIsolationPFChargedHadrons_ = pfTmp;
  selectedIsolationPFNeutralHadrons_ = pfTmp;
  selectedIsolationPFGammaCandidates_ = pfTmp;

  selectedPFChargedHadrons_ = pfTmp;
  selectedPFNeutralHadrons_ = pfTmp;
  selectedPFGammaCandidates_ = pfTmp;

  maximumHcalTowerEnergy_ = -1000;
    transverseIpSignificance_leadTk_ =-1000.;
    mass_ = -1000; 
     trackerMass_ = -1000;
     sumPtIsolation_=-1000;
     emOverHadronEnergy_=-1000;
     emIsolation_=-1000;
}



Tau * Tau::clone() const {
  return new Tau( * this );
}

bool Tau::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( track(), o->track() ))
	   );
}

