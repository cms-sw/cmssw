//
// $Id: Jet.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Jet_h
#define DataFormats_PatCandidates_Jet_h

/**
  \class    Jet Jet.h "DataFormats/PatCandidates/interface/Jet.h"
  \brief    Analysis-level calorimeter jet class

   Jet implements the analysis-level calorimeter jet class within the
   'pat' namespace

  \author   Steven Lowette
  \version  $Id: Jet.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  typedef reco::CaloJet JetType;


  class Jet : public PATObject<JetType> {

    friend class PATJetProducer;

    public:

      Jet();
      Jet(const JetType & aJet);
      virtual ~Jet();

      reco::Particle  getGenParton() const;
      reco::GenJet    getGenJet() const;
      int             getPartonFlavour() const;
      float           getNoCorrF() const;
      float           getUdsCorrF() const;
      float           getGluCorrF() const;
      float           getCCorrF() const;
      float           getBCorrF() const;
      JetType         getRecJet() const;
      Jet             getNoCorrJet() const;
      Jet             getUdsCorrJet() const;
      Jet             getGluCorrJet() const;
      Jet             getCCorrJet() const;
      Jet             getBCorrJet() const;
      Jet             getMCFlavCorrJet() const;
      Jet             getWCorrJet() const;
      float           getBDiscriminator(std::string theLabel) const;
      reco::JetTagRef getBJetTagRef(std::string theLabel) const;
      void            dumpBTagLabels() const;
      float           getLRPhysicsJetVar(unsigned int i) const;
      float           getLRPhysicsJetVal(unsigned int i) const;
      float           getLRPhysicsJetLRval() const;
      float           getLRPhysicsJetProb() const;
      float           getJetCharge() const;
      const reco::TrackRefVector &
                      getAssociatedTracks() const;

    protected:

      void            setGenParton(const reco::Particle & gp);
      void            setGenJet(const reco::GenJet & gj);
      void            setPartonFlavour(int jetFl);
      void            setScaleCalibFactors(float noCorrF, float udsCorrF, float gCorrF, float cCorrF, float bCorrF);
      void            setBResolutions(float bResET_, float bResEta_, float bResPhi_, float bResA_, float bResB_, float bResC_, float bResD_, float bResTheta_);
      void            addBDiscriminatorPair(std::pair<std::string, float> & thePair);
      void            addBJetTagRefPair(std::pair<std::string, reco::JetTagRef> & thePair);
      void            setLRPhysicsJetVarVal(const std::vector<std::pair<float, float> > & varValVec);
      void            setLRPhysicsJetLRval(float clr);
      void            setLRPhysicsJetProb(float plr);

    protected:

      // MC info
      std::vector<reco::Particle> genParton_;
      std::vector<reco::GenJet>   genJet_;
      int jetFlavour_;
      // energy scale correction factors
      // WARNING! noCorrF brings you back to the uncorrected jet, the other
      //          factors bring you from uncorrected to the desired correction
      float noCorrF_, udsCorrF_, gCorrF_, cCorrF_, bCorrF_;
      // additional resolutions for the b-jet hypothesis
      float bResET_, bResEta_, bResPhi_, bResA_, bResB_, bResC_, bResD_, bResTheta_;
      std::vector<float> bCovM_;
      // b-tag related members
      std::vector<std::pair<std::string, float> >          pairDiscriVector_;
      std::vector<std::pair<std::string, reco::JetTagRef> > pairJetTagRefVector_;
      // jet cleaning members (not used yet)
      std::vector<std::pair<float, float> > lrPhysicsJetVarVal_;
      float lrPhysicsJetLRval_;
      float lrPhysicsJetProb_;
      // jet charge members
      float  jetCharge_;
      reco::TrackRefVector associatedTracks_;

  };


}

#endif
