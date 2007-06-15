//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopJet.h,v 1.5 2007/06/15 16:47:45 heyninck Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.5 2007/06/15 16:47:45 heyninck Exp $
*/


#include "DataFormats/JetReco/interface/CaloJet.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


typedef reco::CaloJet JetType;

class TopJet : public TopObject<JetType>
{
   
   public:
      TopJet();
      TopJet(JetType);
      virtual ~TopJet();
            
      void 		setGenJet(Particle);
      void    		setRecJet(JetType);
      void    		setFitJet(TopParticle);
       void    		setBdiscriminant(double);
      void 		setLRPhysicsJetVarVal(std::vector<std::pair<double, double> >);
      void 		setLRPhysicsJetLRval(double);
      void 		setLRPhysicsJetProb(double);
      void              addBdiscriminantPair(std::pair<std::string,double>);
      void              addBJetTagRefPair(std::pair<std::string, reco::JetTagRef>);
      void              setQuarkFlavour(int);
      
      
      reco::Particle	getGenJet() const;
      JetType 		getRecJet() const;
      TopParticle  	getFitJet() const;
      double  		getBdiscriminant() const;
      double            getLRPhysicsJetVar(unsigned int i) const;
      double            getLRPhysicsJetVal(unsigned int i) const;
      double            getLRPhysicsJetLRval() const;
      double            getLRPhysicsJetProb() const;
      double            getBdiscriminantFromPair(std::string) const;
      reco::JetTagRef   getBJetTagRefFromPair(std::string) const;
      void              dumpBTagLabels() const;
      double            getQuarkFlavour() const;
 
   protected:
      reco::Particle    genJet;
      JetType     	recJet;
      TopParticle 	fitJet;
      int               jetFlavour;
      double      	bdiscr, lrPhysicsJetLRval, lrPhysicsJetProb;     
      std::vector<std::pair<double, double> >               lrPhysicsJetVarVal;
      std::vector<std::pair<std::string, double> >          pairDiscriVector; 
      std::vector<std::pair<std::string, reco::JetTagRef> > pairDiscriJetTagRef;



};

#endif
