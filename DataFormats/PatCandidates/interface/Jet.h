//
// $Id: Jet.h,v 1.2 2008/01/16 20:33:20 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Jet_h
#define DataFormats_PatCandidates_Jet_h

/**
  \class    Jet Jet.h "DataFormats/PatCandidates/interface/Jet.h"
  \brief    Analysis-level calorimeter jet class

   Jet implements the analysis-level calorimeter jet class within the
   'pat' namespace

  \author   Steven Lowette
  \version  $Id: Jet.h,v 1.2 2008/01/16 20:33:20 lowette Exp $
*/

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  typedef reco::CaloJet JetType;


  class Jet : public PATObject<JetType> {

    public:

      Jet();
      Jet(const JetType & aJet);
      virtual ~Jet();

      const reco::Particle * genParton() const;
      const reco::GenJet *   genJet() const;
      int             partonFlavour() const;
      float           noCorrF() const;
      float           udsCorrF() const;
      float           gluCorrF() const;
      float           cCorrF() const;
      float           bCorrF() const;
      JetType         recJet() const;
      Jet             noCorrJet() const;
      Jet             udsCorrJet() const;
      Jet             gluCorrJet() const;
      Jet             cCorrJet() const;
      Jet             bCorrJet() const;
      Jet             mcFlavCorrJet() const;
      Jet             wCorrJet() const;
      float           bDiscriminator(std::string theLabel) const;
      reco::JetTagRef bJetTagRef(std::string theLabel) const;
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
      void            setScaleCalibFactors(float noCorrF, float udsCorrF, float gluCorrF, float cCorrF, float bCorrF);
      void            setBResolutions(float bResET_, float bResEta_, float bResPhi_, float bResA_, float bResB_, float bResC_, float bResD_, float bResTheta_);
      void            addBDiscriminatorPair(std::pair<std::string, float> & thePair);
      void            addBJetTagRefPair(std::pair<std::string, reco::JetTagRef> & thePair);
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
      // WARNING! noCorrF brings you back to the uncorrected jet, the other
      //          factors bring you from uncorrected to the desired correction
      float noCorrF_, udsCorrF_, gluCorrF_, cCorrF_, bCorrF_;
      // additional resolutions for the b-jet hypothesis
      float bResET_, bResEta_, bResPhi_, bResA_, bResB_, bResC_, bResD_, bResTheta_;
      std::vector<float> bCovM_;
      // b-tag related members
      std::vector<std::pair<std::string, float> >           pairDiscriVector_;
      std::vector<std::pair<std::string, reco::JetTagRef> > pairJetTagRefVector_;
      // jet cleaning members (not used yet)
      std::vector<std::pair<float, float> > lrPhysicsJetVarVal_;
      float lrPhysicsJetLRval_;
      float lrPhysicsJetProb_;
      // jet charge members
      float jetCharge_;

  };


}

#endif
