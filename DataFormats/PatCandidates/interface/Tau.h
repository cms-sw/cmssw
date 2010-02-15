//
// $Id: Tau.h,v 1.24 2009/06/15 08:28:48 veelken Exp $
//

#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h

/**
  \class    pat::Tau Tau.h "DataFormats/PatCandidates/interface/Tau.h"
  \brief    Analysis-level tau class

   pat::Tau implements the analysis-level tau class within the 'pat' namespace.
   It inherits from reco::BaseTau, copies all the information from the source
   reco::CaloTau or reco::PFTau, and adds some PAT-specific variable.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Christophe Delaere, Giovanni Petrucciani, Frederic Ronga, Colin Bernet
  \version  $Id: Tau.h,v 1.24 2009/06/15 08:28:48 veelken Exp $
*/


#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/Common/interface/BoolCache.h"

#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"
#include "DataFormats/PatCandidates/interface/TauCaloSpecific.h"


// Define typedefs for convenience
namespace pat {
  class Tau;
  typedef std::vector<Tau>              TauCollection; 
  typedef edm::Ref<TauCollection>       TauRef; 
  typedef edm::RefVector<TauCollection> TauRefVector; 
}


// Class definition
namespace pat {

  class Tau : public Lepton<reco::BaseTau> {

    public:

      typedef std::pair<std::string,float> IdPair;

      /// default constructor
      Tau();
      /// constructor from a reco tau
      Tau(const reco::BaseTau & aTau);
      /// constructor from a RefToBase to a reco tau (to be superseded by Ptr counterpart)
      Tau(const edm::RefToBase<reco::BaseTau> & aTauRef);
      /// constructor from a Ptr to a reco tau
      Tau(const edm::Ptr<reco::BaseTau> & aTauRef);
      /// destructor
      virtual ~Tau();

      /// required reimplementation of the Candidate's clone method
      virtual Tau * clone() const { return new Tau(*this); }

      // ---- methods for content embedding ----
      /// override the reco::BaseTau::isolationTracks method, to access the internal storage of the isolation tracks
      const reco::TrackRefVector & isolationTracks() const;
      /// override the reco::BaseTau::leadTrack method, to access the internal storage of the leading track
      reco::TrackRef leadTrack() const;
      /// override the reco::BaseTau::signalTracks method, to access the internal storage of the signal tracks
	const reco::TrackRefVector & signalTracks() const;	
      /// method to store the isolation tracks internally
      void embedIsolationTracks();
      /// method to store the leading track internally
      void embedLeadTrack();
      /// method to store the signal tracks internally
      void embedSignalTracks();

      // ---- matched GenJet methods ----
      /// return matched GenJet, built from the visible particles of a generated tau
      const reco::GenJet * genJet() const;
      /// set the matched GenJet
      void setGenJet(const reco::GenJetRef & ref);

      // ---- CaloTau accessors (getters only) ----
      /// Returns true if this pat::Tau was made from a reco::CaloTau
      bool isCaloTau() const { return !caloSpecific_.empty(); }
      /// return CaloTau info or throw exception 'not CaloTau'
      const pat::tau::TauCaloSpecific & caloSpecific() const ;
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

      // ---- PFTau accessors (getters only) ----
      /// Returns true if this pat::Tau was made from a reco::PFTau
      bool isPFTau() const { return !pfSpecific_.empty(); }
      /// return PFTau info or throw exception 'not PFTau'
      const pat::tau::TauPFSpecific & pfSpecific() const ;
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
      const reco::PFCandidateRef & leadPFNeutralCand() const { return pfSpecific().leadPFNeutralCand_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidateRef & leadPFCand() const { return pfSpecific().leadPFCand_; }
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

      /// reconstructed tau decay mode (specific to PFTau)
      int decayMode() const { return pfSpecific().decayMode_; }
      /// set decay mode
      void setDecayMode(int);

      // ---- methods for tau ID ----
      /// Returns a specific tau ID associated to the pat::Tau given its name
      /// For cut-based IDs, the value is 1.0 for good, 0.0 for bad.
      /// The names are defined within the configuration parameterset "tauIDSources"
      /// in PhysicsTools/PatAlgos/python/producersLayer1/tauProducer_cfi.py .
      /// Note: an exception is thrown if the specified ID is not available
      float tauID(const std::string & name) const;
      /// Returns true if a specific ID is available in this pat::Tau
      bool isTauIDAvailable(const std::string & name) const;
      /// Returns all the tau IDs in the form of <name,value> pairs
      /// The 'default' ID is the first in the list
      const std::vector<IdPair> &  tauIDs() const { return tauIDs_; }
      /// Store multiple tau ID values, discarding existing ones
      /// The first one in the list becomes the 'default' tau id 
      void setTauIDs(const std::vector<IdPair> & ids) { tauIDs_ = ids; }

    protected:

      // ---- for content embedding ----
      bool embeddedIsolationTracks_;
      std::vector<reco::Track> isolationTracks_;
      mutable reco::TrackRefVector isolationTracksTransientRefVector_;
      mutable edm::BoolCache       isolationTracksTransientRefVectorFixed_;
      bool embeddedLeadTrack_;
      std::vector<reco::Track> leadTrack_;
      bool embeddedSignalTracks_;
      std::vector<reco::Track> signalTracks_;
      mutable reco::TrackRefVector signalTracksTransientRefVector_;
      mutable edm::BoolCache       signalTracksTransientRefVectorFixed_;
      // ---- matched GenJet holder ----
      std::vector<reco::GenJet> genJet_;
      // ---- tau ID's holder ----
      std::vector<IdPair> tauIDs_;
      // ---- CaloTau specific variables  ----
      /// holder for CaloTau info, or empty vector if PFTau
      std::vector<pat::tau::TauCaloSpecific> caloSpecific_;
      // ---- PFTau specific variables  ----
      /// holder for PFTau info, or empty vector if CaloTau
      std::vector<pat::tau::TauPFSpecific> pfSpecific_;
  };
}

#endif
