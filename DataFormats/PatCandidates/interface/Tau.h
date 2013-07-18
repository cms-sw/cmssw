//
// $Id: Tau.h,v 1.34.2.2 2013/05/31 14:55:50 veelken Exp $
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
  \version  $Id: Tau.h,v 1.34.2.2 2013/05/31 14:55:50 veelken Exp $
*/


#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"
#include "DataFormats/PatCandidates/interface/TauCaloSpecific.h"
#include "DataFormats/PatCandidates/interface/TauJetCorrFactors.h"

// Define typedefs for convenience
namespace pat {
  class Tau;
  typedef std::vector<Tau>              TauCollection; 
  typedef edm::Ref<TauCollection>       TauRef; 
  typedef edm::RefVector<TauCollection> TauRefVector; 
}

namespace reco {
  /// pipe operator (introduced to use pat::Tau with PFTopProjectors)
  std::ostream& operator<<(std::ostream& out, const pat::Tau& obj);
}

// Class definition
namespace pat {

  class Tau : public Lepton<reco::BaseTau> {
    /// make friends with PATTauProducer so that it can set the initial
    /// jet energy scale unequal to raw calling the private initializeJEC
    /// function, which should be non accessible to any other user
    friend class PATTauProducer;

    public:

      typedef std::pair<std::string, float> IdPair;

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
      ///- PFTau specific content -
      /// method to store the leading candidate internally
      void embedLeadPFCand();
      /// method to store the leading charged hadron candidate internally
      void embedLeadPFChargedHadrCand();
      /// method to store the leading neutral candidate internally
      void embedLeadPFNeutralCand();
      /// method to store the signal candidates internally
      void embedSignalPFCands();
      /// method to store the signal charged hadrons candidates internally
      void embedSignalPFChargedHadrCands();
      /// method to store the signal neutral hadrons candidates internally
      void embedSignalPFNeutralHadrCands();
      /// method to store the signal gamma candidates internally
      void embedSignalPFGammaCands();
      /// method to store the isolation candidates internally
      void embedIsolationPFCands();
      /// method to store the isolation charged hadrons candidates internally
      void embedIsolationPFChargedHadrCands();
      /// method to store the isolation neutral hadrons candidates internally
      void embedIsolationPFNeutralHadrCands();
      /// method to store the isolation gamma candidates internally
      void embedIsolationPFGammaCands();
      
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
      const reco::PFJetRef & pfJetRef() const { return pfSpecific().pfJetRef_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      reco::PFRecoTauChargedHadronRef leadTauChargedHadronCandidate() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidatePtr leadPFChargedHadrCand() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float leadPFChargedHadrCandsignedSipt() const { return pfSpecific().leadPFChargedHadrCandsignedSipt_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidatePtr leadPFNeutralCand() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const reco::PFCandidatePtr leadPFCand() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& signalPFCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& signalPFChargedHadrCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& signalPFNeutrHadrCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& signalPFGammaCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFRecoTauChargedHadron> & signalTauChargedHadronCandidates() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::RecoTauPiZero> & signalPiZeroCandidates() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& isolationPFCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& isolationPFChargedHadrCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& isolationPFNeutrHadrCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFCandidatePtr>& isolationPFGammaCands() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::PFRecoTauChargedHadron> & isolationTauChargedHadronCandidates() const;
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      const std::vector<reco::RecoTauPiZero> & isolationPiZeroCandidates() const;
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
      bool electronPreIDDecision() const { return pfSpecific().electronPreIDDecision_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float caloComp() const { return pfSpecific().caloComp_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      float segComp() const { return pfSpecific().segComp_; }
      /// Method copied from reco::PFTau. 
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau
      bool muonDecision() const { return pfSpecific().muonDecision_; }
      
      /// ---- Tau lifetime information ----
      /// Filled from PFTauTIPAssociation.
      /// Throws an exception if this pat::Tau was not made from a reco::PFTau	
      const reco::PFTauTransverseImpactParameter::Point& dxy_PCA() const { return pfSpecific().dxy_PCA_; }
      double dxy() const { return pfSpecific().dxy_; }
      double dxy_error() const { return pfSpecific().dxy_error_; }
      double dxy_Sig() const;
      const reco::VertexRef& primaryVertex() const { return pfSpecific().pv_; }
      const reco::PFTauTransverseImpactParameter::Point& primaryVertexPos() const { return pfSpecific().pvPos_; }
      const reco::PFTauTransverseImpactParameter::CovMatrix& primaryVertexCov() const { return pfSpecific().pvCov_; }
      bool hasSecondaryVertex() const { return pfSpecific().hasSV_; }
      const reco::PFTauTransverseImpactParameter::Vector& flightLength() const { return pfSpecific().flightLength_; } 
      double flightLengthSig() const { return pfSpecific().flightLengthSig_; }
      reco::PFTauTransverseImpactParameter::CovMatrix flightLengthCov() const;
      const reco::VertexRef& secondaryVertex() const { return pfSpecific().sv_; }
      const reco::PFTauTransverseImpactParameter::Point& secondaryVertexPos() const { return pfSpecific().svPos_; }
      const reco::PFTauTransverseImpactParameter::CovMatrix& secondaryVertexCov() const { return pfSpecific().svCov_; }

      /// Methods copied from reco::Jet.
      /// (accessible from reco::CaloTau/reco::PFTau via reco::CaloTauTagInfo/reco::PFTauTagInfo)
      const reco::Candidate::LorentzVector& p4Jet() const;
      float etaetaMoment() const;
      float phiphiMoment() const;
      float etaphiMoment() const;

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
      float tauID(const char* name ) const {return tauID( std::string(name) );}
      /// Returns true if a specific ID is available in this pat::Tau
      bool isTauIDAvailable(const std::string & name) const;
      /// Returns all the tau IDs in the form of <name,value> pairs
      /// The 'default' ID is the first in the list
      const std::vector<IdPair> &  tauIDs() const { return tauIDs_; }
      /// Store multiple tau ID values, discarding existing ones
      /// The first one in the list becomes the 'default' tau id 
      void setTauIDs(const std::vector<IdPair> & ids) { tauIDs_ = ids; }

      /// pipe operator (introduced to use pat::Tau with PFTopProjectors)
      friend std::ostream& reco::operator<<(std::ostream& out, const Tau& obj);

      /// ---- methods for jet corrections ----
      /// returns the labels of all available sets of jet energy corrections
      const std::vector<std::string> availableJECSets() const;
      // returns the available JEC Levels for a given jecSet
      const std::vector<std::string> availableJECLevels(const int& set = 0) const;
      // returns the available JEC Levels for a given jecSet
      const std::vector<std::string> availableJECLevels(const std::string& set) const { return availableJECLevels(jecSet(set)); };
      /// returns true if the jet carries jet energy correction information
      /// at all
      bool jecSetsAvailable() const { return !jec_.empty(); }
      /// returns true if the jet carries a set of jet energy correction
      /// factors with the given label
      bool jecSetAvailable(const std::string& set) const {return (jecSet(set) >= 0); };
      /// returns true if the jet carries a set of jet energy correction
      /// factors with the given label
      bool jecSetAvailable(const unsigned int& set) const {return (set < jec_.size()); };
      /// returns the label of the current set of jet energy corrections
      std::string currentJECSet() const { 
	return currentJECSet_<jec_.size() ? jec_.at(currentJECSet_).jecSet() : std::string("ERROR"); 
      }
      /// return the name of the current step of jet energy corrections
      std::string currentJECLevel() const { 
	return currentJECSet_<jec_.size() ? jec_.at(currentJECSet_).jecLevel(currentJECLevel_) : std::string("ERROR"); 
      }
      /// correction factor to the given level for a specific set
      /// of correction factors, starting from the current level
      float jecFactor(const std::string& level, const std::string& set = "") const;
      /// correction factor to the given level for a specific set
      /// of correction factors, starting from the current level
      float jecFactor(const unsigned int& level, const unsigned int& set = 0) const;
      /// copy of the jet corrected up to the given level for the set
      /// of jet energy correction factors, which is currently in use
      Tau correctedTauJet(const std::string& level, const std::string& set = "") const;
      /// copy of the jet corrected up to the given level for the set
      /// of jet energy correction factors, which is currently in use
      Tau correctedTauJet(const unsigned int& level, const unsigned int& set = 0) const;
      /// p4 of the jet corrected up to the given level for the set
      /// of jet energy correction factors, which is currently in use
      const LorentzVector& correctedP4(const std::string& level, const std::string& set = "") const { 
	return correctedTauJet(level, set).p4(); 
      }
      /// p4 of the jet corrected up to the given level for the set
      /// of jet energy correction factors, which is currently in use
      const LorentzVector& correctedP4(const unsigned int& level, const unsigned int& set = 0) const { 
	return correctedTauJet(level, set).p4(); 
      }

    protected:
   
      /// index of the set of jec factors with given label; returns -1 if no set
      /// of jec factors exists with the given label
      int jecSet(const std::string& label) const;
      /// update the current JEC set; used by correctedJet
      void currentJECSet(const unsigned int& set) { currentJECSet_=set; };
      /// update the current JEC level; used by correctedJet
      void currentJECLevel(const unsigned int& level) { currentJECLevel_=level; };
      /// add more sets of energy correction factors
      void addJECFactors(const TauJetCorrFactors& jec) {jec_.push_back(jec); };
      /// initialize the jet to a given JEC level during creation starting from Uncorrected
      void initializeJEC(unsigned int level, const unsigned int set = 0);

      // ---- for content embedding ----
      bool embeddedIsolationTracks_;
      std::vector<reco::Track> isolationTracks_;
      mutable reco::TrackRefVector isolationTracksTransientRefVector_;
      mutable bool isolationTracksTransientRefVectorFixed_;
      bool embeddedLeadTrack_;
      std::vector<reco::Track> leadTrack_;
      bool embeddedSignalTracks_;
      std::vector<reco::Track> signalTracks_;
      mutable reco::TrackRefVector signalTracksTransientRefVector_;
      mutable bool signalTracksTransientRefVectorFixed_;
      // specific for PFTau
      std::vector<reco::PFCandidate> leadPFCand_;
      bool embeddedLeadPFCand_;
      std::vector<reco::PFCandidate> leadPFChargedHadrCand_;
      bool embeddedLeadPFChargedHadrCand_;
      std::vector<reco::PFCandidate> leadPFNeutralCand_;
      bool embeddedLeadPFNeutralCand_;

      std::vector<reco::PFCandidate> signalPFCands_;
      bool embeddedSignalPFCands_;
      mutable std::vector<reco::PFCandidatePtr> signalPFCandsTransientPtrs_;
      mutable bool signalPFCandsPtrsFixed_;
      std::vector<reco::PFCandidate> signalPFChargedHadrCands_;
      bool embeddedSignalPFChargedHadrCands_;
      mutable std::vector<reco::PFCandidatePtr> signalPFChargedHadrCandsTransientPtrs_;
      mutable bool signalPFChargedHadrCandsPtrsFixed_;
      std::vector<reco::PFCandidate> signalPFNeutralHadrCands_;
      bool embeddedSignalPFNeutralHadrCands_;
      mutable std::vector<reco::PFCandidatePtr> signalPFNeutralHadrCandsTransientPtrs_;
      mutable bool signalPFNeutralHadrCandsPtrsFixed_;
      std::vector<reco::PFCandidate> signalPFGammaCands_;
      bool embeddedSignalPFGammaCands_;
      mutable std::vector<reco::PFCandidatePtr> signalPFGammaCandsTransientPtrs_;
      mutable bool signalPFGammaCandsPtrsFixed_;
      std::vector<reco::PFCandidate> isolationPFCands_;
      bool embeddedIsolationPFCands_;
      mutable std::vector<reco::PFCandidatePtr> isolationPFCandsTransientPtrs_;
      mutable bool isolationPFCandsPtrsFixed_;
      std::vector<reco::PFCandidate> isolationPFChargedHadrCands_;
      bool embeddedIsolationPFChargedHadrCands_;
      mutable std::vector<reco::PFCandidatePtr> isolationPFChargedHadrCandsTransientPtrs_;
      mutable bool isolationPFChargedHadrCandsPtrsFixed_;
      std::vector<reco::PFCandidate> isolationPFNeutralHadrCands_;
      bool embeddedIsolationPFNeutralHadrCands_;
      mutable std::vector<reco::PFCandidatePtr> isolationPFNeutralHadrCandsTransientPtrs_;
      mutable bool isolationPFNeutralHadrCandsPtrsFixed_;
      std::vector<reco::PFCandidate> isolationPFGammaCands_;
      bool embeddedIsolationPFGammaCands_;
      mutable std::vector<reco::PFCandidatePtr> isolationPFGammaCandsTransientPtrs_;
      mutable bool isolationPFGammaCandsPtrsFixed_;

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

      // ---- energy scale correction factors ----
      // energy scale correction factors; the string carries a potential label if
      // more then one set of correction factors is embedded. The label corresponds
      // to the label of the jetCorrFactors module that has been embedded.
      std::vector<pat::TauJetCorrFactors> jec_;
      // currently applied set of jet energy correction factors (i.e. the index in
      // jetEnergyCorrections_)
      unsigned int currentJECSet_;
      // currently applied jet energy correction level
      unsigned int currentJECLevel_;
  };
}

#endif
