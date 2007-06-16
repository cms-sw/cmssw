//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopJet.h,v 1.6 2007/06/15 23:58:07 lowette Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.6 2007/06/15 23:58:07 lowette Exp $
*/


#include "DataFormats/JetReco/interface/CaloJet.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


typedef reco::CaloJet JetType;


class TopJet : public TopObject<JetType> {

   friend class TopJetProducer;
   
   public:

      TopJet();
      TopJet(JetType);
      virtual ~TopJet();
            
      reco::Particle	getGenJet() const;
      JetType 		getRecJet() const;
      TopParticle  	getFitJet() const;
      double  		getBDiscriminator() const;
      double            getLRPhysicsJetVar(unsigned int i) const;
      double            getLRPhysicsJetVal(unsigned int i) const;
      double            getLRPhysicsJetLRval() const;
      double            getLRPhysicsJetProb() const;
      double            getBDiscriminator(std::string theLabel) const;
      reco::JetTagRef   getBJetTagRef(std::string theLabel) const;
      void              dumpBTagLabels() const;
      double            getPartonFlavour() const;
 
      void    		setFitJet(TopParticle fj);
      void 		setLRPhysicsJetVarVal(std::vector<std::pair<double, double> >);
      void 		setLRPhysicsJetLRval(double clr);
      void 		setLRPhysicsJetProb(double plr);
      
   protected:

      void 		setGenJet(reco::Particle gj);
      void    		setRecJet(JetType rj);
      void    	        setBDiscriminator(double);
      void              addBDiscriminatorPair(std::pair<std::string,double>);
      void              addBJetTagRefPair(std::pair<std::string, reco::JetTagRef>);
      void              setPartonFlavour(int jetf);

   protected:

      reco::Particle    genJet_;
      JetType     	recJet_;
      TopParticle 	fitJet_;
      int               jetFlavour_;
      double      	bDiscr_, lrPhysicsJetLRval_, lrPhysicsJetProb_;
      std::vector<std::pair<double, double> >               lrPhysicsJetVarVal_;
      std::vector<std::pair<std::string, double> >          pairDiscriVector_;
      std::vector<std::pair<std::string, reco::JetTagRef> > pairJetTagRefVector_;

};


#endif
