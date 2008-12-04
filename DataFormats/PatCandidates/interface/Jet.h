//
// $Id: Jet.h,v 1.27 2008/10/09 17:48:23 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Jet_h
#define DataFormats_PatCandidates_Jet_h

/**
  \class    pat::Jet Jet.h "DataFormats/PatCandidates/interface/Jet.h"
  \brief    Analysis-level calorimeter jet class

   Jet implements the analysis-level calorimeter jet class within the
   'pat' namespace

  \author   Steven Lowette
  \version  $Id: Jet.h,v 1.27 2008/10/09 17:48:23 lowette Exp $
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

  typedef reco::Jet JetType;
  typedef reco::CaloJet::Specific CaloSpecific;
  typedef reco::PFJet::Specific PFSpecific;

  class Jet : public PATObject<JetType> {

    public:

      /// default constructor
      Jet();
      /// constructor from a JetType
      Jet(const JetType & aJet);
      /// constructor from ref to JetType
      Jet(const edm::RefToBase<JetType> & aJetRef);
      /// constructor from ref to JetType
      Jet(const edm::Ptr<JetType> & aJetRef);
      /// destructor
      virtual ~Jet();

      virtual Jet * clone() const { return new Jet(*this); }

      /// return the matched generated parton
      const reco::GenParticle * genParton() const { return genParticle(); }
      /// return the matched generated jet
      const reco::GenJet * genJet() const;
      /// return the flavour of the parton underlying the jet
      int partonFlavour() const;

      // Return true if this jet carries jet energy correction information
      bool  hasJetCorrFactors() const { return !jetEnergyCorrections_.empty(); }
      /// return the correction factor for this jet. Throws an exception if they're not available.
      const JetCorrFactors & jetCorrFactors() const {
        if (!hasJetCorrFactors()) throw cms::Exception("Not Available") << "This pat::Jet does not carry jet energy correction information.\n";
        return jetEnergyCorrections_.front();
      }
      /// Return the current level of jet energy corrections
      std::string jetCorrName() const { 
	return jetCorrFactors().corrStep(jetCorrStep()); 
      }
      /// Return flavour of the current level of jet energy corrections
      std::string jetCorrFlavour() const {
	return jetCorrFactors().flavour(jetCorrStep()); 
      }
      /// Return the current level of jet energy corrections
      JetCorrFactors::CorrStep jetCorrStep() const { //FIXME: this one should be private
        return jetEnergyCorrectionStep_;
      }
      /// Total correction factor to target step, starting from jetCorrStep()
      float jetCorrFactor(std::string &step, const std::string &flavour="") const {
	return jetCorrFactors().correction(jetCorrFactors().corrStep(step, flavour), jetCorrStep());
      }
      /// Copy of this jet with correction factor to target step
      Jet correctedJet(const std::string &step, const std::string &flavour="") const ;
      /// method to set the energy scale correction factors
      void setJetCorrFactors(const JetCorrFactors & jetCorrF);
      /// method to set the energy scale correction step used to make this jet
      void setJetCorrStep(JetCorrFactors::CorrStep step); //FIXME: this one shouls be private
#ifdef TO_BE_ADDED_LATER
      /// Return true if this jet carries the jet correction factors of a different set, for systematic studies
      bool hasCorrFactors(const std::string &set) const ;
      /// Return the jet correction factors of a different set, for systematic studies
      const JetCorrFactors & jetCorrFactors(const std::string &set) const ;
      /// Return the jet correction factor from a different set, relative to the present energy corrections
      float jetCorrFactor(const std::string &set, JetCorrFactors::CorrStep target) const ;
      /// Copy of this jet with correction factor to target step from a different set
      Jet correctedJet(const std::string &set, JetCorrFactors::CorrStep target) const ;
      /// return the correction factor for this jet
      float jetCorrFactor(const std::string &step, const std::string &flavour) const ;
      /// return a copy of this jet with the requested correction factor applied
      Jet   correctedJet( const std::string &step, const std::string &flavour) const ;
      /// method to set the energy scale correction factors from a different set
      void setJetCorrFactors(const std::string &set, const JetCorrFactors & jetCorrF);
#endif

      /// get b discriminant from label name
      float bDiscriminator(const std::string &theLabel) const;

      /// get vector of paire labelname-disciValue
      const std::vector<std::pair<std::string, float> > & getPairDiscri() const;

      /// Get a tagInfo with the given name, or NULL if none is found. 
      /// You should omit the 'TagInfos' part from the label
      const reco::BaseTagInfo            * tagInfo(const std::string &label) const;
      /// Get a tagInfo with the given name and type or NULL if none is found. 
      /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
      /// You should omit the 'TagInfos' part from the label
      const reco::TrackIPTagInfo         * tagInfoTrackIP(const std::string &label="") const;
      /// Get a tagInfo with the given name and type or NULL if none is found. 
      /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
      /// You should omit the 'TagInfos' part from the label
      const reco::SoftLeptonTagInfo      * tagInfoSoftLepton(const std::string &label="") const;
      /// Get a tagInfo with the given name and type or NULL if none is found. 
      /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
      /// You should omit the 'TagInfos' part from the label
      const reco::SecondaryVertexTagInfo * tagInfoSecondaryVertex(const std::string &label="") const;
      /// Sets a tagInfo with the given name from an edm::Ptr<T> to it. 
      /// If the label ends with 'TagInfos', the 'TagInfos' is stripped out.
      void  addTagInfo(const std::string &label, 
                       const edm::Ptr<reco::BaseTagInfo> &info) ;

      /// method to return the JetCharge computed when creating the Jet
      float jetCharge() const;
      /// method to return a vector of refs to the tracks associated to this jet
      const reco::TrackRefVector & associatedTracks() const;

      /// method to store the CaloJet constituents internally
      void setCaloTowers(const std::vector<CaloTowerPtr> & caloTowers);
      /// method to set the matched parton
      void setGenParton(const reco::GenParticleRef & gp, bool embed=false) { setGenParticleRef(gp, embed); }
      /// method to set the matched generated jet
      void setGenJet(const reco::GenJet & gj);
      /// method to set the flavour of the parton underlying the jet
      void setPartonFlavour(int partonFl);
//      /// method to set the energy scale correction factors
//      void setJetCorrFactors(const JetCorrFactors & jetCorrF);
//      /// method to set correction factor to go back to an uncorrected jet
//      void setNoCorrFactor(float noCorrF);
      /// method to add a algolabel-discriminator pair
      void addBDiscriminatorPair(const std::pair<std::string, float> & thePair);

      /// method to set the jet charge
      void setJetCharge(float jetCharge);

      /// method to set the vector of refs to the tracks associated to this jet
      void setAssociatedTracks(const reco::TrackRefVector &tracks);

      bool isCaloJet()  const { return !specificCalo_.empty(); }
      bool isPFJet()    const { return !specificPF_.empty(); }
      bool isBasicJet() const { return !(isCaloJet() || isPFJet()); }

      const CaloSpecific & caloSpecific() const { 
          if (specificCalo_.empty()) throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a CaloJet.\n";
          return specificCalo_[0];
      }
      const PFSpecific & pfSpecific() const { 
          if (specificPF_.empty()) throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a PFJet.\n";
          return specificPF_[0];
      }

      //================== Calo Jet specific information ====================
      /** Returns the maximum energy deposited in ECAL towers*/
      float maxEInEmTowers() const {return caloSpecific().mMaxEInEmTowers;}
      /** Returns the maximum energy deposited in HCAL towers*/
      float maxEInHadTowers() const {return caloSpecific().mMaxEInHadTowers;}
      /** Returns the jet hadronic energy fraction*/
      float energyFractionHadronic () const {return caloSpecific().mEnergyFractionHadronic;}
      /** Returns the jet electromagnetic energy fraction*/
      float emEnergyFraction() const {return caloSpecific().mEnergyFractionEm;}
      /** Returns the jet hadronic energy in HB*/
      float hadEnergyInHB() const {return caloSpecific().mHadEnergyInHB;}
      /** Returns the jet hadronic energy in HO*/
      float hadEnergyInHO() const {return caloSpecific().mHadEnergyInHO;}
      /** Returns the jet hadronic energy in HE*/
      float hadEnergyInHE() const {return caloSpecific().mHadEnergyInHE;}
      /** Returns the jet hadronic energy in HF*/
      float hadEnergyInHF() const {return caloSpecific().mHadEnergyInHF;}
      /** Returns the jet electromagnetic energy in EB*/
      float emEnergyInEB() const {return caloSpecific().mEmEnergyInEB;}
      /** Returns the jet electromagnetic energy in EE*/
      float emEnergyInEE() const {return caloSpecific().mEmEnergyInEE;}
      /** Returns the jet electromagnetic energy extracted from HF*/
      float emEnergyInHF() const {return caloSpecific().mEmEnergyInHF;}
      /** Returns area of contributing towers */
      float towersArea() const {return caloSpecific().mTowersArea;}
      /** Returns the number of constituents carrying a 90% of the total Jet energy*/
      int n90() const {return nCarrying (0.9);}
      /** Returns the number of constituents carrying a 60% of the total Jet energy*/
      int n60() const {return nCarrying (0.6);}

      /// convert generic constituent to specific type
      //      static CaloTowerPtr caloTower (const reco::Candidate* fConstituent);
      /// Get specific constituent of the CaloJet. 
      /// If the caloTowers were embedded, this reference is transient only and must not be persisted
      CaloTowerPtr getCaloConstituent (unsigned fIndex) const;
      /// Get the constituents of the CaloJet. 
      /// If the caloTowers were embedded, these reference are transient only and must not be persisted
      std::vector<CaloTowerPtr> getCaloConstituents () const;

    //================== PF Jet specific information ====================
      /// chargedHadronEnergy
      float chargedHadronEnergy () const {return pfSpecific().mChargedHadronEnergy;}
      ///  chargedHadronEnergyFraction
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

      /// Get a pointer to a Candididate constituent of the jet 
      /// Needs to be re-implemented because of CaloTower embedding
      virtual const reco::Candidate * daughter(size_t i) const {
          return (embeddedCaloTowers_ ?  &caloTowers_[i] : reco::Jet::daughter(i));
      }
      /// Get the number of constituents 
      /// Needs to be re-implemented because of CaloTower embedding
      virtual size_t numberOfDaughters() const {
          return (embeddedCaloTowers_ ? caloTowers_.size() : reco::Jet::numberOfDaughters() );
      }
 
    protected:

      // information originally in external branches
      bool embeddedCaloTowers_;
      CaloTowerCollection caloTowers_;
      // MC info
      std::vector<reco::GenJet> genJet_;
      int partonFlavour_;
      // energy scale correction factors

      pat::JetCorrFactors::CorrStep    jetEnergyCorrectionStep_;
      std::vector<pat::JetCorrFactors> jetEnergyCorrections_;
#if TO_BE_ADDED_LATER
      // Names for the additional jet energy corrections for systematic studies
      // The default one carries no name, to save disk space.
      // This means jetEnergyCorrectionExtraNames_.size() < jetEnergyCorrections_.size()
      std::vector<std::string>         jetEnergyCorrectionExtraNames_;
#endif

      // b-tag related members
      std::vector<std::pair<std::string, float> >           pairDiscriVector_;
      // track association
      reco::TrackRefVector associatedTracks_;
      // jet charge members
      float jetCharge_;

      std::vector<std::string>          tagInfoLabels_;
      edm::OwnVector<reco::BaseTagInfo> tagInfos_;  
      template<typename T> const T * tagInfoByType() const ; 

      std::vector<CaloSpecific> specificCalo_;
      std::vector<PFSpecific>   specificPF_;
      void tryImportSpecific(const JetType &source);

#if TO_BE_PORTED_LATER
    static const std::string correctionNames_[NrOfCorrections];
#endif
  };


}

#endif
