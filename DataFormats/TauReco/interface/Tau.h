#ifndef TauReco_Tau_h
#define TauReco_Tau_h

/* class Tau
 * short base class for persistent Tau object 
 * Tau is a candidate based class with extra information needed for the Tau Tagging
 * author Simone Gennai
 * created: Jun 21 2007,
 * revised: Jul 23 2007
 */

#include "DataFormats/TauReco/interface/TauFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <limits>

namespace reco {
  class Tau: public RecoCandidate {
  public:
    Tau();
    Tau(Charge q, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
    virtual ~Tau(){}
    Tau* clone()const{return new Tau(*this);}

    const CaloJetRef& getcalojetRef()const{return CaloJetRef_;}
    void setcalojetRef(const CaloJetRef x) {CaloJetRef_=x;}
   
    const PFJetRef& getpfjetRef()const{return PFJetRef_;}
    void setpfjetRef(const PFJetRef x) {PFJetRef_=x;}
   
    // reference to the leadTrack
    const TrackRef& getleadTrack() const {return leadTrack_;}
    // get the signed transverse impact parameter significance of the lead Track
    float getleadTracksignedSipt()const{return leadTracksignedSipt_;}
    
    // reference to Track reconstructed in the signal Cone
    const TrackRefVector& getSignalTracks() const {return signalTracks_;}

    // reference to the Track in Isolation annulus
    const TrackRefVector& getIsolationTracks() const {return isolationTracks_;}

    // references to the selectedTracks (all the tracks passing quality cuts)
    const TrackRefVector& getSelectedTracks() const {return selectedTracks_;}

    const PFCandidateRef& getleadPFChargedHadrCand() const {return leadPFChargedHadrCand_;}   
    // get the signed transverse impact parameter significance of the Track constituting the lead charged hadron PFCandidate 
    float getleadPFChargedHadrCandsignedSipt()const{return leadPFChargedHadrCandsignedSipt_;}

    const PFCandidateRefVector& getSignalPFChargedHadrCands() const {return selectedSignalPFChargedHadrCands_;}
    const PFCandidateRefVector& getSignalPFNeutrHadrCands() const {return selectedSignalPFNeutrHadrCands_;}
    const PFCandidateRefVector& getSignalPFGammaCands() const {return selectedSignalPFGammaCands_;}
    const PFCandidateRefVector& getIsolationPFChargedHadrCands() const {return selectedIsolationPFChargedHadrCands_;}
    const PFCandidateRefVector& getIsolationPFNeutrHadrCands() const {return selectedIsolationPFNeutrHadrCands_;}
    const PFCandidateRefVector& getIsolationPFGammaCands() const {return selectedIsolationPFGammaCands_;}

    const PFCandidateRefVector& getSelectedPFCands() const {return selectedPFCands_;}
    const PFCandidateRefVector& getSelectedPFChargedHadrCands() const {return selectedPFChargedHadrCands_;}
    const PFCandidateRefVector& getSelectedPFNeutrHadrCands() const {return selectedPFNeutrHadrCands_;}
    const PFCandidateRefVector& getSelectedPFGammaCands() const {return selectedPFGammaCands_;}
    
    // get the invariantMass
    float getInvariantMass() const { return mass_;}

    // get invariantMass with tracks only 
    //float getTksInvariantMass() const {return trackerMass_;}

    // get the sum of the Pt of the  isolation Annulus tracks
    float getSumPtIsolation() const {return sumPtIsolation_;}
    // get the ratio EM energy / Hadron energy
    float getEmEnergyFraction() const { return  emOverHadronEnergy_;}
    // get maximum Hcal tower energy
    float getMaximumHcalEnergy() const { return maximumHcalTowerEnergy_;}
    // get em isolation variable
    float getEMIsolation() const { return emIsolation_;}
    // get the number of Ecal clusters used for mass tag
    int getNumberOfEcalClusters() const  { return numberOfEcalClusters_ ;}

    void setleadTrack(const TrackRef& myTrack) { leadTrack_ = myTrack;}
    void setleadTracksignedSipt(const float& x){leadTracksignedSipt_=x;}
    
    void setSignalTracks(const TrackRefVector& myTracks)  { signalTracks_ = myTracks;}
    void setIsolationTracks(const TrackRefVector& myTracks)  { isolationTracks_ = myTracks;}
    void setSelectedTracks(const TrackRefVector& myTracks)  {selectedTracks_ =myTracks;}

    void setleadPFChargedHadrCand(const PFCandidateRef& myLead) { leadPFChargedHadrCand_=myLead;}   
    void setleadPFChargedHadrCandsignedSipt(const float& x){leadPFChargedHadrCandsignedSipt_=x;}
    
    void setSignalPFChargedHadrCands(const PFCandidateRefVector& myParts)  { selectedSignalPFChargedHadrCands_ = myParts;}
    void setIsolationPFChargedHadrCands(const PFCandidateRefVector& myParts)  { selectedIsolationPFChargedHadrCands_ = myParts;}
    void setSignalPFNeutrHadrCands(const PFCandidateRefVector& myParts)  { selectedSignalPFNeutrHadrCands_ = myParts;}
    void setIsolationPFNeutrHadrCands(const PFCandidateRefVector& myParts)  { selectedIsolationPFNeutrHadrCands_ = myParts;}
    void setSignalPFGammaCands( const PFCandidateRefVector& myParts)  { selectedSignalPFGammaCands_ = myParts;}
    void setIsolationPFGammaCands( const PFCandidateRefVector& myParts)  { selectedIsolationPFGammaCands_ = myParts;}

    void setSelectedPFCands( const PFCandidateRefVector& myParts)  { selectedPFCands_ = myParts;}
    void setSelectedPFChargedHadrCands(const PFCandidateRefVector& myParts)  { selectedPFChargedHadrCands_ = myParts;}
    void setSelectedPFNeutrHadrCands(const PFCandidateRefVector& myParts)  { selectedPFNeutrHadrCands_ = myParts;}
    void setSelectedPFGammaCands( const PFCandidateRefVector& myParts)  { selectedPFGammaCands_ = myParts;}

    void setInvariantMass(const float& mass)  {  mass_= mass;}

    //set invariantMass with tracks only 
    //void setTksInvariantMass()  { trackerMass_;}

    void setSumPtIsolation(const float& sumPt)  { sumPtIsolation_ = sumPt;}
    void setEmEnergyFraction(const float& emOverH)  {   emOverHadronEnergy_ = emOverH;}
    void setMaximumHcalEnergy(const float& maxHcal)  {  maximumHcalTowerEnergy_ = maxHcal;}
    void setEMIsolation(const float& emIso)  {  emIsolation_ = emIso;}
    void setNumberOfEcalClusters(const int& myClus)  { numberOfEcalClusters_ = myClus;}
  private:
    /// check overlap with another candidate
    virtual bool overlap(const Candidate& theCand)const{
      const RecoCandidate* theRecoCand=dynamic_cast<const RecoCandidate *>(&theCand);
      return (theRecoCand!=0 && (checkOverlap(track(),theRecoCand->track())));
    }
    CaloJetRef CaloJetRef_;
    PFJetRef PFJetRef_;
    TrackRef leadTrack_;
    TrackRefVector signalTracks_, isolationTracks_;
    TrackRefVector selectedTracks_;

    PFCandidateRef leadPFChargedHadrCand_;
    PFCandidateRefVector selectedPFCands_,selectedPFChargedHadrCands_,selectedPFNeutrHadrCands_,selectedPFGammaCands_;
    PFCandidateRefVector selectedSignalPFChargedHadrCands_, selectedSignalPFNeutrHadrCands_, selectedSignalPFGammaCands_;
    PFCandidateRefVector selectedIsolationPFChargedHadrCands_, selectedIsolationPFNeutrHadrCands_, selectedIsolationPFGammaCands_;

    float maximumHcalTowerEnergy_;
    float leadPFChargedHadrCandsignedSipt_;
    float leadTracksignedSipt_;
    float mass_; 
    float trackerMass_;
    float sumPtIsolation_;
    float emOverHadronEnergy_;
    float emIsolation_;
    int numberOfEcalClusters_;
  };
}
#endif
