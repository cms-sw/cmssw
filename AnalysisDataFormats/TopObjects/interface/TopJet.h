//
// $Id: TopJet.h,v 1.10 2007/06/30 14:42:54 gpetrucc Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.10 2007/06/30 14:42:54 gpetrucc Exp $
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
    TopJetType      getRecJet() const;
    int             getPartonFlavour() const;
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
    void            setRecJet(const TopJetType & rj);
    void            setPartonFlavour(int jetFl);
    void            addBDiscriminatorPair(std::pair<std::string,double> & thePair);
    void            addBJetTagRefPair(std::pair<std::string, reco::JetTagRef> & thePair);
    void            setLRPhysicsJetVarVal(const std::vector<std::pair<double, double> > & varValVec);
    void            setLRPhysicsJetLRval(double clr);
    void            setLRPhysicsJetProb(double plr);

  protected:

    std::vector<reco::Particle>                genParton_;
    // FIXME: add parton jet
    std::vector<reco::GenJet>                  genJet_;
    std::vector<reco::Particle::LorentzVector> recJet_;
    // b-tag related members
    int jetFlavour_;
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
