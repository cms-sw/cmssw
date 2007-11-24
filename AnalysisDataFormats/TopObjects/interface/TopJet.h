//
// $Id: TopJet.h,v 1.11 2007/07/05 23:33:34 lowette Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.11 2007/07/05 23:33:34 lowette Exp $
*/


#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


typedef reco::CaloJet TopJetType;


class TopJet : public TopObject<TopJetType> {

  friend class TopJetProducer;

  public:

    TopJet();
    TopJet(TopJetType);
    virtual ~TopJet();

    reco::Particle  getGenParton() const;
    // FIXME: add parton jet
    reco::GenJet    getGenJet() const;
    int             getPartonFlavour() const;
    float           getNoCorrF() const;
    float           getUdsCorrF() const;
    float           getGluCorrF() const;
    float           getCCorrF() const;
    float           getBCorrF() const;
    TopJetType      getRecJet() const;
    TopJet          getNoCorrJet() const;
    TopJet          getUdsCorrJet() const;
    TopJet          getGluCorrJet() const;
    TopJet          getCCorrJet() const;
    TopJet          getBCorrJet() const;
    TopJet          getMCFlavCorrJet() const;
    TopJet          getWCorrJet() const;
    double          getBDiscriminator(std::string theLabel) const;
    reco::JetTagRef getBJetTagRef(std::string theLabel) const;
    void            dumpBTagLabels() const;
    double          getLRPhysicsJetVar(unsigned int i) const;
    double          getLRPhysicsJetVal(unsigned int i) const;
    double          getLRPhysicsJetLRval() const;
    double          getLRPhysicsJetProb() const;

    float                        getJetCharge() const ;
    const reco::TrackRefVector&  getAssociatedTracks() const ;
   
  protected:

    void            setGenParton(const reco::Particle & gp);
    // FIXME: add parton jet
    void            setGenJet(const reco::GenJet & gj);
    void            setPartonFlavour(int jetFl);
    void            setScaleCalibFactors(float noCorrF, float udsCorrF, float gCorrF, float cCorrF, float bCorrF);
    void            setBResolutions(float bResET_, float bResEta_, float bResPhi_, float bResA_, float bResB_, float bResC_, float bResD_, float bResTheta_);
    void            addBDiscriminatorPair(std::pair<std::string,double> & thePair);
    void            addBJetTagRefPair(std::pair<std::string, reco::JetTagRef> & thePair);
    void            setLRPhysicsJetVarVal(const std::vector<std::pair<double, double> > & varValVec);
    void            setLRPhysicsJetLRval(double clr);
    void            setLRPhysicsJetProb(double plr);

  protected:

    // MC info
    std::vector<reco::Particle> genParton_;
    // FIXME: add parton jet
    std::vector<reco::GenJet>   genJet_;
    int jetFlavour_;
    // energy scale correction factors
    // WARNING! noCorrF brings you back to the uncorrected jet, the other
    //          factors bring you from uncorrected to the desired correction
    float noCorrF_, udsCorrF_, gCorrF_, cCorrF_, bCorrF_;
    // additional resolutions for the b-jet hypothesis
    float bResET_, bResEta_, bResPhi_, bResA_, bResB_, bResC_, bResD_, bResTheta_;
    std::vector<double> bCovM_;
    // b-tag related members
    std::vector<std::pair<std::string, double> >          pairDiscriVector_;
    std::vector<std::pair<std::string, reco::JetTagRef> > pairJetTagRefVector_;
    // jet cleaning members
    std::vector<std::pair<double, double> > lrPhysicsJetVarVal_;
    double lrPhysicsJetLRval_;
    double lrPhysicsJetProb_;
    // jet charge members
    float  jetCharge_;
    reco::TrackRefVector associatedTracks_;

};


#endif
