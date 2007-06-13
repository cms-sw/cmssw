//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopJet.h,v 1.3 2007/05/23 09:00:14 heyninck Exp $
//

#ifndef TopObjects_TopJet_h
#define TopObjects_TopJet_h

/**
  \class    TopJet TopJet.h "AnalysisDataFormats/TopObjects/interface/TopJet.h"
  \brief    High-level top jet container

   TopJet contains a jet as a TopObject

  \author   Steven Lowette
  \version  $Id: TopJet.h,v 1.3 2007/05/23 09:00:14 heyninck Exp $
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
      void              AddBdiscriminantPair(pair<string,double>);
      void              AddBJetTagRefPair(pair<string,JetTagRef>);
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
      void              DumpLabel() const;
      double            getQuarkFlavour() const;
 
   protected:
      Particle   	genJet;
      JetType     	recJet;
      TopParticle 	fitJet;
      int               jetFlavour;
      double      	bdiscr, lrPhysicsJetLRval, lrPhysicsJetProb;     
      std::vector<std::pair<double, double> > lrPhysicsJetVarVal;
      vector<pair<string,double> >            PairDiscriVector; 
      vector<pair<string,JetTagRef> >         PairDiscriJetTagRef; 



};

#endif
