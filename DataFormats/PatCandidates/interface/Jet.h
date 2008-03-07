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

      Jet();
      Jet(const JetType & aJet);
      Jet(const edm::RefToBase<JetType> & aJetRef);
      virtual ~Jet();

      const reco::Particle * genParton() const;
      const reco::GenJet *   genJet() const;
      int             partonFlavour() const;
      JetCorrFactors  jetCorrFactors() const;
      JetType         recJet() const;
      Jet             noCorrJet() const;
      Jet             defaultCorrJet() const;
      Jet             udsCorrJet() const;
      Jet             gluCorrJet() const;
      Jet             cCorrJet() const;
      Jet             bCorrJet() const;
      Jet             mcFlavCorrJet() const;
      Jet             wCorrJet() const;
      float           bDiscriminator(std::string theLabel) const;
      const std::vector<reco::TrackIPTagInfoRef> 
                      bTagIPTagInfoRef()  const;
      const std::vector<reco::SoftLeptonTagInfoRef>
                      bTagSoftLeptonERef() const;
      const std::vector<reco::SoftLeptonTagInfoRef>
                      bTagSoftLeptonMRef() const;
      const std::vector<reco::SecondaryVertexTagInfoRef>
                      bTagSecondaryVertexTagInfoRef() const;
      float           lrPhysicsJetVar(unsigned int i) const;
      float           lrPhysicsJetVal(unsigned int i) const;
      float           lrPhysicsJetLRval() const;
      float           lrPhysicsJetProb() const;
      float           jetCharge() const;
      const reco::TrackRefVector &
                      associatedTracks() const;

      void            setGenParton(const reco::Particle & gp);
      void            setGenJet(const reco::GenJet & gj);
      void            setPartonFlavour(int partonFl);
      void            setJetCorrFactors(const JetCorrFactors & jetCorrF);
      void            setNoCorrFactor(float noCorrF);
      void            setBResolutions(float bResEt_, float bResEta_, float bResPhi_, float bResA_, float bResB_, float bResC_, float bResD_, float bResTheta_);
      void            addBDiscriminatorPair(std::pair<std::string, float> & thePair);
      void            addBTagIPTagInfoRef(const reco::TrackIPTagInfoRef & tagRef);
      void            addBTagSoftLeptonERef(const reco::SoftLeptonTagInfoRef & tagRef);
      void            addBTagSoftLeptonMRef(const reco::SoftLeptonTagInfoRef & tagRef);
      void            addBTagSecondaryVertexTagInfoRef(const reco::SecondaryVertexTagInfoRef & tagRef);
      void            setLRPhysicsJetVarVal(const std::vector<std::pair<float, float> > & varValVec);
      void            setLRPhysicsJetLRval(float clr);
      void            setLRPhysicsJetProb(float plr);
      void            setJetCharge(float jetCharge);
      
      
  public:

      reco::TrackRefVector associatedTracks_;

    protected:

      // MC info
      std::vector<reco::Particle> genParton_;
      std::vector<reco::GenJet>   genJet_;
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
