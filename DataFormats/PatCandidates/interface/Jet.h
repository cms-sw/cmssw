//
// $Id: Jet.h,v 1.36 2009/07/16 09:28:08 rwolf Exp $
//

#ifndef DataFormats_PatCandidates_Jet_h
#define DataFormats_PatCandidates_Jet_h

/**
  \class    pat::Jet Jet.h "DataFormats/PatCandidates/interface/Jet.h"
  \brief    Analysis-level calorimeter jet class

   Jet implements the analysis-level calorimeter jet class within the
   'pat' namespace

  \author   Steven Lowette, Giovanni Petrucciani, Roger Wolf, Christian Autermann
  \version  $Id: Jet.h,v 1.36 2009/07/16 09:28:08 rwolf Exp $
*/


#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/OwnVector.h"


// Define typedefs for convenience
namespace pat {
  class Jet;
  typedef std::vector<Jet>              JetCollection; 
  typedef edm::Ref<JetCollection>       JetRef; 
  typedef edm::RefVector<JetCollection> JetRefVector; 
}


// Class definition
namespace pat {

  typedef reco::CaloJet::Specific CaloSpecific;
  typedef reco::PFJet::Specific PFSpecific;

  class Jet : public PATObject<reco::Jet> {

    public:

      /// default constructor
      Jet();
      /// constructor from a reco::Jet
      Jet(const reco::Jet & aJet);
      /// constructor from ref to reco::Jet
      Jet(const edm::RefToBase<reco::Jet> & aJetRef);
      /// constructor from ref to reco::Jet
      Jet(const edm::Ptr<reco::Jet> & aJetRef);
      /// destructor
      virtual ~Jet();
      /// required reimplementation of the Candidate's clone method
      virtual Jet * clone() const { return new Jet(*this); }

      /// ---- methods for MC matching ----

      /// return the matched generated parton
      const reco::GenParticle * genParton() const { return genParticle(); }
      /// return the matched generated jet
      const reco::GenJet * genJet() const;
      /// return the flavour of the parton underlying the jet
      int partonFlavour() const;

      /// ---- methods for jet corrections ----

      /// return true if the jet carries jet energy correction information
      bool  hasCorrFactors() const { return !jetEnergyCorrections_.empty(); }
      /// return true if the jet carries the jet correction factors of a different set, for systematic studies
      bool  hasCorrFactorSet(const std::string& set) const;
      /// return the label of the current set of jet energy corrections
      std::string corrFactorSetLabel() const { return corrFactors_()->getLabel(); }
      /// return label-names of all available sets of jet energy corrections
      const std::vector<std::string> corrFactorSetLabels() const;
      /// return the name of the current step of jet energy corrections
      std::string corrStep() const;
      /// return flavour of the current step of jet energy corrections
      std::string corrFlavour() const;
      /// total correction factor to target step, starting from jetCorrStep(),
      /// for the set of correction factors, which is currently in use
      float corrFactor(const std::string& step, const std::string& flavour="") const;
      /// total correction factor to target step, starting from jetCorrStep(),
      /// for a specific set of correction factors
      float corrFactor(const std::string& step, const std::string& flavour, const std::string& set) const;
      /// copy of the jet with correction factor to target step for
      /// the set of correction factors, which is currently in use 
      Jet correctedJet(const JetCorrFactors::CorrStep& step) const;
      /// copy of the jet with correction factor to target step for
      /// the set of correction factors, which is currently in use 
      Jet correctedJet(const std::string& step, const std::string& flavour="") const;
      /// copy of this jet with correction factor to target step
      /// for a specific set of correction factors
      Jet correctedJet(const JetCorrFactors::CorrStep& step, const std::string& set) const;
      /// copy of this jet with correction factor to target step
      /// for a specific set of correction factors
      Jet correctedJet(const std::string& step, const std::string& flavour, const std::string& set) const;
      /// p4 of the jet with correction factor to target step for
      /// the set of correction factors, which is currently in use 
      const LorentzVector& correctedP4(const JetCorrFactors::CorrStep& step) const { return correctedJet(step).p4(); };
      /// p4 of the jet with correction factor to target step for
      /// the set of correction factors, which is currently in use 
      const LorentzVector& correctedP4(const std::string& step, const std::string& flavour="") const { return correctedJet(step, flavour).p4(); };
      /// p4 of the jet with correction factor to target step for
      /// the set of correction factors, which is currently in use 
      const LorentzVector& correctedP4(const JetCorrFactors::CorrStep& step, const std::string& set) const { return correctedJet(step, set).p4(); };
      /// p4 of the jet with correction factor to target step for
      /// the set of correction factors, which is currently in use 
      const LorentzVector& correctedP4(const std::string& step, const std::string& flavour, const std::string& set) const { return correctedJet(step, flavour, set).p4(); };
      /// method to set the energy scale correction factors this will change the jet's momentum! 
      /// it should only be used by the PATJetProducer; per default the first element in 
      /// jetEnergyCorrections_ is taken into consideration
      void setCorrStep(JetCorrFactors::CorrStep step);
      /// to be used by PATJetProducer: method to set the energy scale correction factors
      void setCorrFactors(const JetCorrFactors & jetCorrF);
      /// to be used by PATJetProducer: method to add more sets of energy scale correction factors
      void addCorrFactors(const JetCorrFactors & jetCorrF);

      /// ---- methods for accessing b-tagging info ----

      /// get b discriminant from label name
      float bDiscriminator(const std::string &theLabel) const;
      /// get vector of paire labelname-disciValue
      const std::vector<std::pair<std::string, float> > & getPairDiscri() const;
      /// get a tagInfo with the given name, or NULL if none is found. 
      /// You should omit the 'TagInfos' part from the label
      const reco::BaseTagInfo            * tagInfo(const std::string &label) const;
      /// get a tagInfo with the given name and type or NULL if none is found. 
      /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
      /// you should omit the 'TagInfos' part from the label
      const reco::TrackIPTagInfo         * tagInfoTrackIP(const std::string &label="") const;
      /// get a tagInfo with the given name and type or NULL if none is found. 
      /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
      /// you should omit the 'TagInfos' part from the label
      const reco::SoftLeptonTagInfo      * tagInfoSoftLepton(const std::string &label="") const;
      /// get a tagInfo with the given name and type or NULL if none is found. 
      /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
      /// you should omit the 'TagInfos' part from the label
      const reco::SecondaryVertexTagInfo * tagInfoSecondaryVertex(const std::string &label="") const;
      /// method to add a algolabel-discriminator pair
      void addBDiscriminatorPair(const std::pair<std::string, float> & thePair);
      /// sets a tagInfo with the given name from an edm::Ptr<T> to it. 
      /// If the label ends with 'TagInfos', the 'TagInfos' is stripped out.
      void  addTagInfo(const std::string &label, 
                       const edm::Ptr<reco::BaseTagInfo> &info) ;

      // ---- track related methods ----

      /// method to return the JetCharge computed when creating the Jet
      float jetCharge() const;
      /// method to return a vector of refs to the tracks associated to this jet
      const reco::TrackRefVector & associatedTracks() const;
      /// method to set the jet charge
      void setJetCharge(float jetCharge);
      /// method to set the vector of refs to the tracks associated to this jet
      void setAssociatedTracks(const reco::TrackRefVector &tracks);

      // ---- methods for content embedding ----

      /// method to store the CaloJet constituents internally
      void setCaloTowers(const std::vector<CaloTowerPtr> & caloTowers);
      /// method to set the matched parton
      void setGenParton(const reco::GenParticleRef & gp, bool embed=false) { setGenParticleRef(gp, embed); }
      /// method to set the matched generated jet
      void setGenJet(const reco::GenJet & gj);
      /// method to set the flavour of the parton underlying the jet
      void setPartonFlavour(int partonFl);


      /// methods for jet ID 
      void setFHPD         (double   fHPD         ){fHPD_ =         fHPD;         }; 
      void setFRBX         (double   fRBX         ){fRBX_ =         fRBX;         }; 
      void setN90Hits      (int      n90Hits      ){n90Hits_ =      n90Hits;      }; 
      void setFSubDetector1(double   fSubDetector1){fSubDetector1_ =fSubDetector1;}; 
      void setFSubDetector2(double   fSubDetector2){fSubDetector2_ =fSubDetector2;}; 
      void setFSubDetector3(double   fSubDetector3){fSubDetector3_ =fSubDetector3;}; 
      void setFSubDetector4(double   fSubDetector4){fSubDetector4_ =fSubDetector4;}; 
      void setRestrictedEMF(double   restrictedEMF){restrictedEMF_ =restrictedEMF;}; 
      void setNHCALTowers  (int      nHCALTowers  ){nHCALTowers_ =  nHCALTowers;  }; 
      void setNECALTowers  (int      nECALTowers  ){nECALTowers_ =  nECALTowers;  };      

      // ---- jet specific methods ----

      /// check to see if the jet is a reco::CaloJet
      bool isCaloJet()  const { return !specificCalo_.empty(); }
      /// check to see if the jet is a reco::PFJet
      bool isPFJet()    const { return !specificPF_.empty(); }
      /// check to see if the jet is no more than a reco::BasicJet
      bool isBasicJet() const { return !(isCaloJet() || isPFJet()); }
      /// retrieve the calo specific part of the jet
      const CaloSpecific & caloSpecific() const { 
          if (specificCalo_.empty()) throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a CaloJet.\n";
          return specificCalo_[0];
      }
      /// retrieve the pf specific part of the jet
      const PFSpecific & pfSpecific() const { 
          if (specificPF_.empty()) throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a PFJet.\n";
          return specificPF_[0];
      }

      // ---- Calo Jet specific information ----

      /// returns the maximum energy deposited in ECAL towers
      float maxEInEmTowers() const {return caloSpecific().mMaxEInEmTowers;}
      /// returns the maximum energy deposited in HCAL towers
      float maxEInHadTowers() const {return caloSpecific().mMaxEInHadTowers;}
      /// returns the jet hadronic energy fraction
      float energyFractionHadronic () const {return caloSpecific().mEnergyFractionHadronic;}
      /// returns the jet electromagnetic energy fraction
      float emEnergyFraction() const {return caloSpecific().mEnergyFractionEm;}
      /// returns the jet hadronic energy in HB
      float hadEnergyInHB() const {return caloSpecific().mHadEnergyInHB;}
      /// returns the jet hadronic energy in HO
      float hadEnergyInHO() const {return caloSpecific().mHadEnergyInHO;}
      /// returns the jet hadronic energy in HE
      float hadEnergyInHE() const {return caloSpecific().mHadEnergyInHE;}
      /// returns the jet hadronic energy in HF
      float hadEnergyInHF() const {return caloSpecific().mHadEnergyInHF;}
      /// returns the jet electromagnetic energy in EB
      float emEnergyInEB() const {return caloSpecific().mEmEnergyInEB;}
      /// returns the jet electromagnetic energy in EE
      float emEnergyInEE() const {return caloSpecific().mEmEnergyInEE;}
      /// returns the jet electromagnetic energy extracted from HF
      float emEnergyInHF() const {return caloSpecific().mEmEnergyInHF;}
      /// returns area of contributing towers
      float towersArea() const {return caloSpecific().mTowersArea;}
      /// returns the number of constituents carrying a 90% of the total Jet energy*/
      int n90() const {return nCarrying (0.9);}
      /// returns the number of constituents carrying a 60% of the total Jet energy*/
      int n60() const {return nCarrying (0.6);}

      /// convert generic constituent to specific type
      //  static CaloTowerPtr caloTower (const reco::Candidate* fConstituent);
      /// get specific constituent of the CaloJet. 
      /// if the caloTowers were embedded, this reference is transient only and must not be persisted
      CaloTowerPtr getCaloConstituent (unsigned fIndex) const;
      /// get the constituents of the CaloJet. 
      /// If the caloTowers were embedded, these reference are transient only and must not be persisted
      std::vector<CaloTowerPtr> getCaloConstituents () const;

      // ---- PF Jet specific information ----

      /// chargedHadronEnergy
      float chargedHadronEnergy () const {return pfSpecific().mChargedHadronEnergy;}
      /// chargedHadronEnergyFraction
      float  chargedHadronEnergyFraction () const {return chargedHadronEnergy () / energy ();}
      /// neutralHadronEnergy
      float neutralHadronEnergy () const {return pfSpecific().mNeutralHadronEnergy;}
      /// neutralHadronEnergyFraction
      float neutralHadronEnergyFraction () const {return neutralHadronEnergy () / energy ();}
      /// chargedEmEnergy
      float chargedEmEnergy () const {return pfSpecific().mChargedEmEnergy;}
      /// chargedEmEnergyFraction
      float chargedEmEnergyFraction () const {return chargedEmEnergy () / energy ();}
      /// chargedMuEnergy
      float chargedMuEnergy () const {return pfSpecific().mChargedMuEnergy;}
      /// chargedMuEnergyFraction
      float chargedMuEnergyFraction () const {return chargedMuEnergy () / energy ();}
      /// neutralEmEnergy
      float neutralEmEnergy () const {return pfSpecific().mNeutralEmEnergy;}
      /// neutralEmEnergyFraction
      float neutralEmEnergyFraction () const {return neutralEmEnergy () / energy ();}
      /// chargedMultiplicity
      float chargedMultiplicity () const {return pfSpecific().mChargedMultiplicity;}
      /// neutralMultiplicity
      float neutralMultiplicity () const {return pfSpecific().mNeutralMultiplicity;}
      /// muonMultiplicity
      float muonMultiplicity () const {return pfSpecific().mMuonMultiplicity;}

      /// convert generic constituent to specific type
      static const reco::PFCandidate* getPFCandidate (const reco::Candidate* fConstituent);
      /// get specific constituent
      const reco::PFCandidate* getPFConstituent (unsigned fIndex) const;
      /// get all constituents
      std::vector <const reco::PFCandidate*> getPFConstituents () const;

      /// get a pointer to a Candididate constituent of the jet 
      /// needs to be re-implemented because of CaloTower embedding
      virtual const reco::Candidate * daughter(size_t i) const {
          return (embeddedCaloTowers_ ?  &caloTowers_[i] : reco::Jet::daughter(i));
      }
      using reco::LeafCandidate::daughter; // avoid hiding the base implementation
      /// get the number of constituents 
      /// needs to be re-implemented because of CaloTower embedding
      virtual size_t numberOfDaughters() const {
          return (embeddedCaloTowers_ ? caloTowers_.size() : reco::Jet::numberOfDaughters() );
      }


      /// accessing Jet ID information
      double fHPD()          const   { return    fHPD_;         }           
      double fRBX()          const   { return    fRBX_;         }           
      int    n90Hits()       const   { return    n90Hits_;      }        
      double fSubDetector1() const   { return    fSubDetector1_;}  
      double fSubDetector2() const   { return    fSubDetector2_;}  
      double fSubDetector3() const   { return    fSubDetector3_;}  
      double fSubDetector4() const   { return    fSubDetector4_;}  
      double restrictedEMF() const   { return    restrictedEMF_;}  
      int    nHCALTowers()   const   { return    nHCALTowers_;  }    
      int    nECALTowers()   const   { return    nECALTowers_;  }       

    protected:

      // ---- for content embedding ----

      bool embeddedCaloTowers_;
      CaloTowerCollection caloTowers_;

      // ---- MC info ----

      std::vector<reco::GenJet> genJet_;
      int partonFlavour_;

      // ---- energy scale correction factors ----

      /// energy scale correction factors
      std::vector<pat::JetCorrFactors> jetEnergyCorrections_; 
      /// the level of the currently applied correction factor
      pat::JetCorrFactors::CorrStep    jetEnergyCorrectionStep_;
      /// index in 'jetEnergyCorrections_' of the currently applied correction factor set
      unsigned activeJetCorrIndex_;

      // ---- b-tag related members ----

      std::vector<std::pair<std::string, float> >           pairDiscriVector_;
      std::vector<std::string>          tagInfoLabels_;
      edm::OwnVector<reco::BaseTagInfo> tagInfos_;  

      // ---- track related members ----

      float jetCharge_;
      reco::TrackRefVector associatedTracks_;

      // ---- specific members ----

      std::vector<CaloSpecific> specificCalo_;
      std::vector<PFSpecific>   specificPF_;

      // ---- id functions ----
      // Mostly english, except for: "f"-fraction, "n"=number, "had"=hadronic, "EM"=electro-magnetic
      double fHPD_;
      double fRBX_;
      int    n90Hits_;
      double fSubDetector1_;
      double fSubDetector2_;
      double fSubDetector3_;
      double fSubDetector4_;
      double restrictedEMF_;
      int    nHCALTowers_;
      int    nECALTowers_;      
      
    private:

      // ---- helper functions ----

      void tryImportSpecific(const reco::Jet &source);
      template<typename T> const T * tagInfoByType() const;

      /// return the jet correction factors of a different set, for systematic studies
      const JetCorrFactors * corrFactors_(const std::string& set) const ;
      /// return the correction factor for this jet. Throws if they're not available.
      const JetCorrFactors * corrFactors_() const;
  };
}

#endif
