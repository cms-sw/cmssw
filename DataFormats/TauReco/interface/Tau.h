#ifndef TauReco_Tau_h
#define TauReco_Tau_h
// \class Tau
// 
// \short base class for persistent Tau object 
// Tau is a candidate based class with extra information needed for the Tau Tagging
// 
//
// \author Simone Gennai
// \version first version on June, 21, 2007
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

namespace reco {
  class Tau: public RecoCandidate {
  public:
    Tau();
    Tau(const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
    Tau * clone () const;

    //Regerence to the leadingTrack
    TrackRef getLeadingTrack() const {return leadingTrack_};
    // reference to Track reconstructed in the signal Cone
    TrackRefVector getSignalTracks() const {return signalTracks_};
    //reference to the Track in Isolation annulus
    TrackRefVector getIsolationTracks() const {return isolationTracks_};
    //references to the selectedTracks (all the tracks passing quality cuts)
    TrackRefVector getSelectedTracks() const {return selectedTracks_};
    //reference to the Charged and Neutral Hadrons and to Gamma candidates
    PFCandidateRefVector getChargedHadrons const {return selectedPFChargedHadrons_};
    PFCandidateRefVector getNeutralHadrons const {return selectedPFNeutralHadrons_};
    PFCandidateRefVector getGammaCandidates const {return selectedPFGammaCandidates__};
    //get the impact parameter of the leading track
    Measurement1D getLeadTkTIP() const {return transverseIp_leadTk};
    Measurement1D getLeadTk3DIP() const {return  3DIp_leadTk_};
    //get the invariantMass
    float getInvariantMass() const { return mass_};

    //get invariantMass with tracks only 
    //float getTksInvariantMass() const {return trackerMass_};

    //get the sum of the Pt of the signal and isolation tracks
    float getSumPtSignalTracks() const {return sumPtSignalTracks_};
    float getSumPtIsolationTracks() const {return sumPtIsolationTracks_};
    //get the ratio EM energy / Hadron energy
    float getEmOverHadronEnergy() const { return  emOverHadronEnergy_};
    //get maximum Hcal tower energy
    float getMaximumHcalTowerEnergy const { return maximumHcalTowerEnergy_};
    //get em isolation variable
    float getEMIsolation() const { return emIsolation_};

  private:
    virtual bool orverlap (const Candidate &) const;
    TrackRef leadingTrack_;
    TrackRefVector signalTracks_, isolationTracks_;
    TrackRefVector selectedTracks_;
    PFCandidateRefVector selectedPFChargedHadrons_, selectedPFNeutralHadrons_, selectedPFGammaCandidates_;

    float maximumHcalTowerEnergy_;
    Measurement1D  transverseIp_leadTk_;
    Measurement1D  3DIp_leadTk_;
    float mass_; 
    float trackerMass_;
    float sumPtSignalTracks_;
    float sumPtIsolationTracks_;
    float emOverHadronEnergy_;
    float emIsolation_;

