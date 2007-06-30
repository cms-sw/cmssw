//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopJet.h,v 1.9 2007/06/23 07:03:21 lowette Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.9 2007/06/23 07:03:21 lowette Exp $
*/


#include "DataFormats/JetReco/interface/CaloJet.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


typedef reco::CaloJet TopJetType;


class TopJet : public TopObject<TopJetType> {

  friend class TopJetProducer;
  friend class TtSemiKinFitterEMom;
  friend class TtSemiKinFitterEtEtaPhi;
  friend class TtSemiKinFitterEtThetaPhi;
  friend class StKinFitterEMom;
  friend class StKinFitterEtEtaPhi;
  friend class StKinFitterEtThetaPhi;

  public:

    TopJet();
    TopJet(TopJetType);
    virtual ~TopJet();

    reco::Particle  getGenParton() const;
    // FIXME: add parton jet
    // FIXME: add GenJet
    TopJetType      getRecJet() const;
    TopParticle     getFitJet() const;
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

    void            setGenParton(reco::Particle gj);
    // FIXME: add parton jet
    // FIXME: add GenJet
    void            setRecJet(TopJetType rj);
    void            setFitJet(TopParticle fj);
    void            setPartonFlavour(int jetFl);
    void            addBDiscriminatorPair(std::pair<std::string,double> thePair);
    void            addBJetTagRefPair(std::pair<std::string, reco::JetTagRef> thePair);
    void            setLRPhysicsJetVarVal(std::vector<std::pair<double, double> > varValVec);
    void            setLRPhysicsJetLRval(double clr);
    void            setLRPhysicsJetProb(double plr);

  protected:

    std::vector<reco::Particle> genParton_;
    // FIXME: add parton jet
    // FIXME: add GenJet
    std::vector<TopJetType>     recJet_;
    std::vector<TopParticle>    fitJet_;
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
