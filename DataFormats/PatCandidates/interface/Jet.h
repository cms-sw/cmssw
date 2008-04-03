//
// $Id$
//

#ifndef DataFormats_PatCandidates_Jet_h
#define DataFormats_PatCandidates_Jet_h

/**
  \class    pat::Jet Jet.h "DataFormats/PatCandidates/interface/Jet.h"
  \brief    Analysis-level calorimeter jet class

   Jet implements the analysis-level calorimeter jet class within the
   'pat' namespace

  \author   Steven Lowette
  \version  $Id$
*/

#include "DataFormats/JetReco/interface/CaloJet.h"
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


namespace pat {


  typedef reco::CaloJet JetType;


  class Jet : public PATObject<JetType> {

    public:

      /// default constructor
      Jet();
      /// constructor from a JetType
      Jet(const JetType & aJet);
      /// constructor from ref to JetType
      Jet(const edm::RefToBase<JetType> & aJetRef);
      /// destructor
      virtual ~Jet();

      /// override the getConstituent method from CaloJet, to access the internal storage of the constituents
      /// this returns a transient Ref which *should never be persisted*!
      CaloTowerRef getConstituent(unsigned int idx) const;
      /// override the getConstituents method from CaloJet, to access the internal storage of the constituents
      /// this returns a transient RefVector which *should never be persisted*!
      std::vector<CaloTowerRef> getConstituents() const;
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
      float bDiscriminator(std::string theLabel) const;
      /// get JetTagRef for TrackIP
      const std::vector<reco::TrackIPTagInfoRef>  bTagIPTagInfoRef() const;
      /// get JetTagRef for electron SoftLepton
      const std::vector<reco::SoftLeptonTagInfoRef> bTagSoftLeptonERef() const;
      /// get JetTagRef for muon SoftLepton
      const std::vector<reco::SoftLeptonTagInfoRef> bTagSoftLeptonMRef() const;
      /// get JetTagRef for SecondaryVertex
      const std::vector<reco::SecondaryVertexTagInfoRef> bTagSecondaryVertexTagInfoRef() const;
      /// get the value of the i'th jet cleaning variable
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
      void setCaloTowers(const std::vector<CaloTowerRef> & caloTowers);
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
      /// method to add a TrackIP TagInfoRef
      void addBTagIPTagInfoRef(const reco::TrackIPTagInfoRef & tagRef);
      /// method to add an electron SoftLepton TagInfoRef
      void addBTagSoftLeptonERef(const reco::SoftLeptonTagInfoRef & tagRef);
      /// method to add a muon SoftLepton TagInfoRef
      void addBTagSoftLeptonMRef(const reco::SoftLeptonTagInfoRef & tagRef);
      /// method to add a SecondaryVertex TagInfoRef
      void addBTagSecondaryVertexTagInfoRef(const reco::SecondaryVertexTagInfoRef & tagRef);
      /// method to set all jet cleaning variable + LR pairs
      void setLRPhysicsJetVarVal(const std::vector<std::pair<float, float> > & varValVec);
      /// method to set the combined jet cleaning likelihood ratio value
      void setLRPhysicsJetLRval(float clr);
      /// method to set the jet cleaning probability
      void setLRPhysicsJetProb(float plr);
      /// method to set the jet charge
      void setJetCharge(float jetCharge);

    public:

      reco::TrackRefVector associatedTracks_;

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
      // jet charge members
      float jetCharge_;
      std::vector<reco::TrackIPTagInfoRef>         bTagIPTagInfoRef_;
      std::vector<reco::SoftLeptonTagInfoRef>      bTagSoftLeptonERef_;
      std::vector<reco::SoftLeptonTagInfoRef>      bTagSoftLeptonMRef_;
      std::vector<reco::SecondaryVertexTagInfoRef> bTagSecondaryVertexTagInfoRef_;

  };


}

#endif
