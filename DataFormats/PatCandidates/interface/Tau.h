//
// $Id: Tau.h,v 1.14 2008/06/09 09:03:19 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h

/**
  \class    pat::Tau Tau.h "DataFormats/PatCandidates/interface/Tau.h"
  \brief    Analysis-level tau class

   Tau implements the analysis-level tau class within the 'pat' namespace.

   It inherits from reco::BaseTau, copies all the information from the source
   reco::CaloTau or reco::PFTau, and adds some PAT-specific variable.


  \author   Steven Lowette, Christophe Delaere, Giovanni Petrucciani
  \version  $Id: Tau.h,v 1.14 2008/06/09 09:03:19 gpetrucc Exp $
*/


#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"
#include "DataFormats/PatCandidates/interface/TauCaloSpecific.h"


namespace pat {


  typedef reco::BaseTau TauType;


  class Tau : public Lepton<TauType> {

    public:

      Tau();
      Tau(const TauType & aTau);
      Tau(const edm::RefToBase<TauType> & aTauRef);
      Tau(const edm::Ptr<TauType> & aTauRef);
      virtual ~Tau();

      virtual Tau * clone() const { return new Tau(*this); }

      /// override the TauType::isolationTracks method, to access the internal storage of the track
      reco::TrackRefVector isolationTracks() const;
      /// override the TauType::track method, to access the internal storage of the track
      reco::TrackRef leadTrack() const;
      /// override the TauType::track method, to access the internal storage of the track
      reco::TrackRefVector signalTracks() const;

      /// method to store the isolation tracks internally
      void embedIsolationTracks();
      /// method to store the isolation tracks internally
      void embedLeadTrack();
      /// method to store the isolation tracks internally
      void embedSignalTracks();

      // ---- CaloTau accessors (getters only) --
      /// Returns true if this pat::Tau was made from a reco::CaloTau
      bool isCaloTau() const { return !caloSpecific_.empty(); }
      // accessors to datamembers
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      reco::CaloTauTagInfoRef caloTauTagInfoRef() const { return caloSpecific().CaloTauTagInfoRef_; }
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float leadTracksignedSipt() const { return caloSpecific().leadTracksignedSipt_; }
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float leadTrackHCAL3x3hitsEtSum() const { return caloSpecific().leadTrackHCAL3x3hitsEtSum_; }
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float leadTrackHCAL3x3hottesthitDEta() const { return caloSpecific().leadTrackHCAL3x3hottesthitDEta_; }
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float signalTracksInvariantMass() const { return caloSpecific().signalTracksInvariantMass_; }
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float TracksInvariantMass() const { return caloSpecific().TracksInvariantMass_; } 
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float isolationTracksPtSum() const { return caloSpecific().isolationTracksPtSum_; }
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float isolationECALhitsEtSum() const { return caloSpecific().isolationECALhitsEtSum_; }
      /// Method copied from reco::CaloTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::CaloTau
      float maximumHCALhitEt() const { return caloSpecific().maximumHCALhitEt_; }

      // ---- PFTau accessors (getters only) --
      /// Returns true if this pat::Tau was made from a reco::PFTau
      bool isPFTau() const { return !pfSpecific_.empty(); }
      //  accessors to datamembers
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFTauTagInfoRef & pfTauTagInfoRef() const { return pfSpecific().PFTauTagInfoRef_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRef & leadPFChargedHadrCand() const { return pfSpecific().leadPFChargedHadrCand_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float leadPFChargedHadrCandsignedSipt() const { return pfSpecific().leadPFChargedHadrCandsignedSipt_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & signalPFCands() const { return pfSpecific().selectedSignalPFCands_; } 
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & signalPFChargedHadrCands() const { return pfSpecific().selectedSignalPFChargedHadrCands_; } 
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & signalPFNeutrHadrCands() const { return pfSpecific().selectedSignalPFNeutrHadrCands_; } 
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & signalPFGammaCands() const { return pfSpecific().selectedSignalPFGammaCands_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & isolationPFCands() const { return pfSpecific().selectedIsolationPFCands_; } 
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & isolationPFChargedHadrCands() const { return pfSpecific().selectedIsolationPFChargedHadrCands_; } 
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & isolationPFNeutrHadrCands() const { return pfSpecific().selectedIsolationPFNeutrHadrCands_; } 
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRefVector & isolationPFGammaCands() const { return pfSpecific().selectedIsolationPFGammaCands_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float isolationPFChargedHadrCandsPtSum() const { return pfSpecific().isolationPFChargedHadrCandsPtSum_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float isolationPFGammaCandsEtSum() const { return pfSpecific().isolationPFGammaCandsEtSum_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float maximumHCALPFClusterEt() const { return pfSpecific().maximumHCALPFClusterEt_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float emFraction() const { return pfSpecific().emFraction_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float hcalTotOverPLead() const { return pfSpecific().hcalTotOverPLead_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float hcalMaxOverPLead() const { return pfSpecific().hcalMaxOverPLead_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float hcal3x3OverPLead() const { return pfSpecific().hcal3x3OverPLead_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float ecalStripSumEOverPLead() const { return pfSpecific().ecalStripSumEOverPLead_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float bremsRecoveryEOverPLead() const { return pfSpecific().bremsRecoveryEOverPLead_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::TrackRef & electronPreIDTrack() const { return pfSpecific().electronPreIDTrack_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float electronPreIDOutput() const { return pfSpecific().electronPreIDOutput_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      bool  electronPreIDDecision() const { return pfSpecific().electronPreIDDecision_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float caloComp() const { return pfSpecific().caloComp_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float segComp() const { return pfSpecific().segComp_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      bool  muonDecision() const { return pfSpecific().muonDecision_; }

    private:
      // ---- Embedding of BaseTau components ----
      bool embeddedIsolationTracks_;
      std::vector<reco::Track> isolationTracks_;
      bool embeddedLeadTrack_;
      std::vector<reco::Track> leadTrack_;
      bool embeddedSignalTracks_;
      std::vector<reco::Track> signalTracks_;

      // ---- CaloTau specific variables  ----
      /// hold the CaloTau info, or empty vector if CaloTau
      std::vector<pat::tau::TauCaloSpecific> caloSpecific_;
      /// return CaloTau info or throw exception 'not CaloTau'
      const pat::tau::TauCaloSpecific & caloSpecific() const ;


      // ---- PFTau specific variables  ----
      /// hold the PFTau info, or empty vector if CaloTau
      std::vector<pat::tau::TauPFSpecific> pfSpecific_;
      /// return PFTau info or throw exception 'not PFTau'
      const pat::tau::TauPFSpecific & pfSpecific() const ;

      // ---- PAT specific variables  ----
      float emEnergyFraction_;
      float eOverP_;
      float leadeOverP_;
      float HhotOverP_;
      float HtotOverP_;

  };


}

#endif
