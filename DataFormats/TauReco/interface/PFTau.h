
#ifndef DataFormats_TauReco_PFTau_h
#define DataFormats_TauReco_PFTau_h

/* class PFTau
 * the object of this class is created by RecoTauTag/RecoTau PFRecoTauProducer EDProducer starting from the PFTauTagInfo object,
 *                          is a hadronic tau-jet candidate -built from a jet made employing a particle flow technique- that analysts manipulate;
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 * created: Jun 21 2007,
 * revised: Sep 12 2007
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <iostream>
#include <limits>


namespace reco {
class PFTau : public BaseTau {
   public:
      PFTau();
      PFTau(Charge q,const LorentzVector &,const Point & = Point( 0, 0, 0 ) );
      virtual ~PFTau() {};
      PFTau* clone()const;

      /// Return a persistent Ref to the PFTauTagInfo used to build this PFTau
      const PFTauTagInfoRef& pfTauTagInfoRef()const;
      void setpfTauTagInfoRef(const PFTauTagInfoRef);

      /// Return a reference to the highest Pt charged hadron candidate
      const PFCandidateRef& leadPFChargedHadrCand()const; 
      /// Return a reference to the highest Pt PFGamma candidate
      const PFCandidateRef& leadPFNeutralCand()const; 
      /// Return the PFTau 'lead candidate', which can be charged or neutral. 
      const PFCandidateRef& leadPFCand()const; 

      void setleadPFChargedHadrCand(const PFCandidateRef&);
      void setleadPFNeutralCand(const PFCandidateRef&);
      void setleadPFCand(const PFCandidateRef&);
      /// signed transverse impact parameter significance of the Track constituting the leading charged hadron PFCandidate 
      float leadPFChargedHadrCandsignedSipt()const;
      void setleadPFChargedHadrCandsignedSipt(const float&);

      ///  PFCandidates which passed quality cuts and are inside a tracker/ECAL/HCAL signal cone around leading charged hadron PFCandidate
      const PFCandidateRefVector& signalPFCands()const;
      void setsignalPFCands(const PFCandidateRefVector&);
      /// List of charged hadrons contained in the signal cone
      const PFCandidateRefVector& signalPFChargedHadrCands()const;
      void setsignalPFChargedHadrCands(const PFCandidateRefVector&);
      /// List of neutral hadrons (HCAL) contained in the signal cone
      const PFCandidateRefVector& signalPFNeutrHadrCands()const;
      void setsignalPFNeutrHadrCands(const PFCandidateRefVector&);
      /// List of gamma candidates (ECAL) contained in the signal cone
      const PFCandidateRefVector& signalPFGammaCands()const;
      void setsignalPFGammaCands(const PFCandidateRefVector&);

      /// PFCandidates which passed quality cuts and are inside a tracker/ECAL/HCAL 
      /// isolation annulus around leading charged hadron PFCandidate
      const PFCandidateRefVector& isolationPFCands()const;
      void setisolationPFCands(const PFCandidateRefVector&);
      const PFCandidateRefVector& isolationPFChargedHadrCands()const;
      void setisolationPFChargedHadrCands(const PFCandidateRefVector&);
      const PFCandidateRefVector& isolationPFNeutrHadrCands()const;
      void setisolationPFNeutrHadrCands(const PFCandidateRefVector&);
      const PFCandidateRefVector& isolationPFGammaCands()const;
      void setisolationPFGammaCands(const PFCandidateRefVector&);

      /// sum of Pt of the charged hadr. PFCandidates inside a tracker isolation annulus 
      /// around leading charged hadron PFCandidate ; NaN if no leading charged hadron PFCandidate
      float isolationPFChargedHadrCandsPtSum()const;
      void setisolationPFChargedHadrCandsPtSum(const float&);

      /// sum of Et of the gamma PFCandidates inside an ECAL isolation annulus around the
      /// leading charged hadron PFCandidate ; NaN if no leading charged hadron PFCandidate 
      float isolationPFGammaCandsEtSum()const;
      void setisolationPFGammaCandsEtSum(const float&);

      /// Et of the highest Et HCAL PFCluster  
      float maximumHCALPFClusterEt()const;
      void setmaximumHCALPFClusterEt(const float&);    

      /// chargedHadronEnergy in signal cone
      float chargedHadronEnergy () const;
      ///  chargedHadronEnergyFraction in signal cone
      float  chargedHadronEnergyFraction () const;
      /// neutralHadronEnergy in signal cone
      float neutralHadronEnergy () const;
      /// neutralHadronEnergyFraction in signal cone
      float neutralHadronEnergyFraction () const;
      /// chargedEmEnergy in signal cone
      float chargedEmEnergy () const;
      /// chargedEmEnergyFraction in signal cone
      float chargedEmEnergyFraction () const;
      /// chargedMuEnergy in signal cone
      float chargedMuEnergy () const;
      /// chargedMuEnergyFraction in signal cone
      float chargedMuEnergyFraction () const;
      /// neutralEmEnergy in signal cone
      float neutralEmEnergy () const;
      /// neutralEmEnergyFraction in signal cone
      float neutralEmEnergyFraction () const;
      /// charged multiplicity in signal cone
      int chargedMultiplicity () const;
      /// neutral multiplicity in signal cone
      int neutralMultiplicity () const;
      /// muon multiplicity in signal cone
      int muonMultiplicity () const;

      void setSpecific(const PFJet::Specific& input);
      const PFJet::Specific& getSpecific () const;

      //Electron rejection
      /// ECAL/HCAL cluster energy
      float emFraction() const; 
      /// total HCAL cluster energy / lead charged hadron momentum
      float hcalTotOverPLead() const; 
      /// max HCAL cluster energy divided by lead charged hadron momentum
      float hcalMaxOverPLead() const; 
      /// Sum of energy in HCAL clusters within R < 0.184 around ECAL impact point of lead track, divided by lead charged hadron momentum
      float hcal3x3OverPLead() const; 
      /// Sum of energy in simple Brem recovery ECAL strip divided by lead charged hadron momentum
      float ecalStripSumEOverPLead() const; 
      /// Brem recovery energy divided by lead charged hadron momentum
      float bremsRecoveryEOverPLead() const; 
      /// Reference to KF track from electron PreID
      reco::TrackRef electronPreIDTrack() const; 
      /// BDT output from electron PreID
      float electronPreIDOutput() const; 
      /// Deciscion computed from electron PreID
      bool electronPreIDDecision() const; 

      void setemFraction(const float&);
      void sethcalTotOverPLead(const float&);
      void sethcalMaxOverPLead(const float&);
      void sethcal3x3OverPLead(const float&);
      void setecalStripSumEOverPLead(const float&);
      void setbremsRecoveryEOverPLead(const float&);
      void setelectronPreIDTrack(const reco::TrackRef&);
      void setelectronPreIDOutput(const float&);
      void setelectronPreIDDecision(const bool&);
      //end of Electron rejection

      // For Muon Rejection
      /// Returns true if lead object has a reference to a reco::Muon
      bool    hasMuonReference()const; 
      float   caloComp()const;
      float   segComp()const;
      bool    muonDecision()const;
      void setCaloComp(const float&);
      void setSegComp(const float&);
      void setMuonDecision(const bool&);

      /// return the number of source Candidates
      /// ( the candidates used to construct this Candidate)
      /// in the case of taus, there is only one source candidate,
      /// which is the corresponding PFJet
      size_type numberOfSourceCandidatePtrs() const {return 1;}

      /// return a RefToBase to the source Candidates
      /// ( the candidates used to construct this Candidate)
      CandidatePtr sourceCandidatePtr( size_type i ) const;

      /// prints information on this PFTau
      void dump(std::ostream& out=std::cout) const;

   private:
      // check overlap with another candidate
      virtual bool overlap(const Candidate&)const;
      PFTauTagInfoRef PFTauTagInfoRef_;
      PFCandidateRef leadPFChargedHadrCand_;
      PFCandidateRef leadPFNeutralCand_, leadPFCand_;
      float leadPFChargedHadrCandsignedSipt_;
      PFCandidateRefVector selectedSignalPFCands_, selectedSignalPFChargedHadrCands_, selectedSignalPFNeutrHadrCands_, selectedSignalPFGammaCands_;
      PFCandidateRefVector selectedIsolationPFCands_, selectedIsolationPFChargedHadrCands_, selectedIsolationPFNeutrHadrCands_, selectedIsolationPFGammaCands_;
      float isolationPFChargedHadrCandsPtSum_;
      float isolationPFGammaCandsEtSum_;
      float maximumHCALPFClusterEt_;

      PFJet::Specific signalConeJetParameters_;

      float emFraction_;
      float hcalTotOverPLead_;
      float hcalMaxOverPLead_;
      float hcal3x3OverPLead_;
      float ecalStripSumEOverPLead_;
      float bremsRecoveryEOverPLead_;
      reco::TrackRef electronPreIDTrack_;
      float electronPreIDOutput_;
      bool electronPreIDDecision_;

      float caloComp_;
      float segComp_;
      bool muonDecision_;
};

std::ostream & operator<<(std::ostream& out, const PFTau& c); 

}
#endif
