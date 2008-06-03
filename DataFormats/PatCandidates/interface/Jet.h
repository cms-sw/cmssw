//
// $Id: Jet.h,v 1.18 2008/05/26 11:22:12 arizzi Exp $
//

#ifndef DataFormats_PatCandidates_Jet_h
#define DataFormats_PatCandidates_Jet_h

/**
  \class    pat::Jet Jet.h "DataFormats/PatCandidates/interface/Jet.h"
  \brief    Analysis-level calorimeter jet class

   Jet implements the analysis-level calorimeter jet class within the
   'pat' namespace

  \author   Steven Lowette
  \version  $Id: Jet.h,v 1.18 2008/05/26 11:22:12 arizzi Exp $
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
//#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
//#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
//#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"

#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include "DataFormats/Common/interface/Ptr.h"

namespace pat {

  typedef reco::Jet JetType;
  typedef reco::CaloJet::Specific CaloSpecific;
  typedef reco::PFJet::Specific PFSpecific;

  class Jet : public PATObject<JetType> {

  public:
    enum CorrectionType { NoCorrection=0, DefaultCorrection,
			  udsCorrection, cCorrection, bCorrection, gCorrection, 
			  NrOfCorrections };

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
      const reco::Particle * genParton() const;
      /// return the matched generated jet
      const reco::GenJet * genJet() const;
      /// return the flavour of the parton underlying the jet
      int partonFlavour() const;
      /// return the correction factor to go to a non-calibrated jet
      JetCorrFactors jetCorrFactors() const;
      /// return the original non-calibrated jet
      JetType recJet() const;
      /// return the associated non-calibrated jet
      Jet noCorrJet() const;
      /// return the associated default-calibrated jet
      Jet defaultCorrJet() const;
      /// return the associated uds-calibrated jet
      Jet udsCorrJet() const;
      /// return the associated gluon-calibrated jet
      Jet gluCorrJet() const;
      /// return the associated c-calibrated jet
      Jet cCorrJet() const;
      /// return the associated b-calibrated jet
      Jet bCorrJet() const;
      /// return the jet calibrated according to the MC flavour truth
      Jet mcFlavCorrJet() const;
      /// return the jet calibrated with weights assuming W decay
      Jet wCorrJet() const;
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

      float lrPhysicsJetVar(unsigned int i) const;
      /// get the likelihood ratio corresponding to the i'th jet cleaning variable
      float lrPhysicsJetVal(unsigned int i) const;
      /// get the overall jet cleaning likelihood ratio
      float lrPhysicsJetLRval() const;
      /// get the overall jet cleaning probability
      float lrPhysicsJetProb() const;
      /// method to return the JetCharge computed when creating the Jet
      float jetCharge() const;
      /// method to return a vector of refs to the tracks associated to this jet
      const reco::TrackRefVector & associatedTracks() const;

      /// method to store the CaloJet constituents internally
      void setCaloTowers(const std::vector<CaloTowerPtr> & caloTowers);
      /// method to set the matched parton
      void setGenParton(const reco::Particle & gp);
      /// method to set the matched generated jet
      void setGenJet(const reco::GenJet & gj);
      /// method to set the flavour of the parton underlying the jet
      void setPartonFlavour(int partonFl);
      /// method to set the energy scale correction factors
      void setJetCorrFactors(const JetCorrFactors & jetCorrF);
      /// method to set correction factor to go back to an uncorrected jet
      void setNoCorrFactor(float noCorrF);
      /// method to set the resolutions under the assumption this is a b-jet
      void setBResolutions(float bResEt_, float bResEta_, float bResPhi_, float bResA_, float bResB_, float bResC_, float bResD_, float bResTheta_);
      /// method to add a algolabel-discriminator pair
      void addBDiscriminatorPair(std::pair<std::string, float> & thePair);

      void setLRPhysicsJetVarVal(const std::vector<std::pair<float, float> > & varValVec);
      /// method to set the combined jet cleaning likelihood ratio value
      void setLRPhysicsJetLRval(float clr);
      /// method to set the jet cleaning probability
      void setLRPhysicsJetProb(float plr);
      /// method to set the jet charge
      void setJetCharge(float jetCharge);
    /// correction factor from correction type
    float correctionFactor (CorrectionType type) const;
    /// auxiliary method to convert a string to a correction type
    static CorrectionType correctionType (const std::string& name);

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

    protected:

      // information originally in external branches
      bool embeddedCaloTowers_;
      CaloTowerCollection caloTowers_;
      // MC info
      std::vector<reco::Particle> genParton_;
      std::vector<reco::GenJet> genJet_;
      int partonFlavour_;
      // energy scale correction factors
      JetCorrFactors jetCorrF_;
      float noCorrF_;
      // additional resolutions for the b-jet hypothesis
      float bResEt_, bResEta_, bResPhi_, bResA_, bResB_, bResC_, bResD_, bResTheta_;
      std::vector<float> bCovM_;
      // b-tag related members
      std::vector<std::pair<std::string, float> >           pairDiscriVector_;
      // jet cleaning members (not used yet)
      std::vector<std::pair<float, float> > lrPhysicsJetVarVal_;
      float lrPhysicsJetLRval_;
      float lrPhysicsJetProb_;
      // track association
      reco::TrackRefVector associatedTracks_;
      // jet charge members
      float jetCharge_;

      std::vector<std::string>          tagInfoLabels_;
      // edm::OwnVector<reco::BaseTagInfo> tagInfos_;  // no, no clone() method :-(
      std::vector<edm::Ptr<reco::BaseTagInfo> > tagInfos_; // cheaper to store than RefToBase
                                                           // not exposed to the user in any case
      template<typename T> const T * tagInfoByType() const ; 

      std::vector<CaloSpecific> specificCalo_;
      std::vector<PFSpecific>   specificPF_;
      void tryImportSpecific(const JetType &source);

    static const std::string correctionNames_[NrOfCorrections];
  };


}

#endif
