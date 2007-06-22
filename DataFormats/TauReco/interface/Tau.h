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
    PFCandidateRefVector getSignalChargedHadrons() const {return selectedSignalPFChargedHadrons_};
    PFCandidateRefVector getSignalNeutralHadrons() const {return selectedSignalPFNeutralHadrons_};
    PFCandidateRefVector getSignalGammaCandidates() const {return selectedSignalPFGammaCandidates__};
    PFCandidateRefVector getIsolationChargedHadrons() const {return selectedIsolationPFChargedHadrons_};
    PFCandidateRefVector getIsolationNeutralHadrons() const {return selectedIsolationPFNeutralHadrons_};
    PFCandidateRefVector getIsolationGammaCandidates() const {return selectedIsolationPFGammaCandidates__};

    //get the impact parameter of the leading track
    Measurement1D getLeadTkTIP() const {return transverseIp_leadTk};
    Measurement1D getLeadTk3DIP() const {return  3DIp_leadTk_};

    //get the invariantMass
    float getInvariantMass() const { return mass_};

    //get invariantMass with tracks only 
    //float getTksInvariantMass() const {return trackerMass_};

    //get the sum of the Pt of the signal and isolation tracks
    float getSumPtSignalCone() const {return sumPtSignal_};
    float getSumPtIsolation() const {return sumPtIsolation_};
    //get the ratio EM energy / Hadron energy
    float getEmOverHadronEnergy() const { return  emOverHadronEnergy_};
    //get maximum Hcal tower energy
    float getMaximumHcalTowerEnergy() const { return maximumHcalTowerEnergy_};
    //get em isolation variable
    float getEMIsolation() const { return emIsolation_};


    void setLeadingTrack(const TrackRef& myTrack) { leadingTrack_ = myTrack};
    void setSignalTracks(const TrackRefVector& myTracks)  { signalTracks_ = myTracks};
    void setIsolationTracks(const TrackRefVector& myTracks)  { isolationTracks_ = myTracks};
    void setSelectedTracks(const TrackRefVector& myTracks)  {selectedTracks_ =myTracks};
    void setSignalChargedHadrons(const FCandidateRefVector& myParts)  { selectedSignalPFChargedHadrons_ = myParts};
    void setIsolationChargedHadrons(const FCandidateRefVector& myParts)  { selectedSignalPFChargedHadrons_ = myParts};
    void setSignalNeutralHadrons(const PFCandidateRefVector& myParts)  { selectedSignalPFNeutralHadrons_ = myParts};
    void setIsolationNeutralHadrons(const PFCandidateRefVector& myParts)  { selectedIsolationPFNeutralHadrons_ = myParts};
    void setSignalGammaCandidates( const PFCandidateRefVector& myParts)  { selectedPFSignalGammaCandidates_ = myParts};
    void setIsolationGammaCandidates( const PFCandidateRefVector& myParts)  { selectedPFIsolationGammaCandidates_ = myParts};
    void setLeadTkTIP(const Measurement1D& myIP)  { transverseIp_leadTk = myIP};
    void setLeadTk3DIP(const Measurement1D& myIP)  {  3DIp_leadTk_=myIP};
    void setInvariantMass(const float& mass)  {  mass_= mass};

    //set invariantMass with tracks only 
    //void setTksInvariantMass()  { trackerMass_};

    void setSumPtSignal(const float& sumPt)  { sumPtSignal_ = sumPt};
    void setSumPtIsolation(const float& sumPt)  { sumPtIsolation_ = sumPt};
    void setEmOverHadronEnergy()const float& emOverH  {   emOverHadronEnergy_ = emOverH};
    void setMaximumHcalTowerEnergy(const float& maxHcal)  {  maximumHcalTowerEnergy_ = maxHcal};
    void setEMIsolation(const float& emIso)  {  emIsolation_ = emIso};


  private:
    virtual bool orverlap (const Candidate &) const;
    TrackRef leadingTrack_;
    TrackRefVector signalTracks_, isolationTracks_;
    TrackRefVector selectedTracks_;
    PFCandidateRefVector selectedSignalPFChargedHadrons_, selectedSignalPFNeutralHadrons_, selectedSignalPFGammaCandidates_;
    PFCandidateRefVector selectedIsolationPFChargedHadrons_, selectedIsolationPFNeutralHadrons_, selectedIsolationPFGammaCandidates_;

    float maximumHcalTowerEnergy_;
    Measurement1D  transverseIp_leadTk_;
    Measurement1D  3DIp_leadTk_;
    float mass_; 
    float trackerMass_;
    float sumPtSignal_;
    float sumPtIsolation_;
    float emOverHadronEnergy_;
    float emIsolation_;

