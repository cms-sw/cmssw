//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopJet.h,v 1.4 2007/06/13 11:33:51 jandrea Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.4 2007/06/13 11:33:51 jandrea Exp $
*/


#include "DataFormats/JetReco/interface/CaloJet.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

using namespace reco;
using namespace std;



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
      void              addBdiscriminantPair(pair<string,double>);
      void              addBJetTagRefPair(pair<string,JetTagRef>);
      void              setQuarkFlavour(int);
      
      
      reco::Particle	getGenJet() const;
      JetType 		getRecJet() const;
      TopParticle  	getFitJet() const;
      double  		getBdiscriminant() const;
      double            getLRPhysicsJetVar(unsigned int i) const;
      double            getLRPhysicsJetVal(unsigned int i) const;
      double            getLRPhysicsJetLRval() const;
      double            getLRPhysicsJetProb() const;
      double            getBdiscriminantFromPair(string) const;
      JetTagRef         getBJetTagRefFromPair(string) const;
      void              dumpBTagLabels() const;
      double            getQuarkFlavour() const;
 
   protected:
      Particle   	genJet;
      JetType     	recJet;
      TopParticle 	fitJet;
      int               jetFlavour;
      double      	bdiscr, lrPhysicsJetLRval, lrPhysicsJetProb;     
      std::vector<std::pair<double, double> > lrPhysicsJetVarVal;
      vector<pair<string,double> >            pairDiscriVector; 
      vector<pair<string,JetTagRef> >         pairDiscriJetTagRef; 



};

#endif
