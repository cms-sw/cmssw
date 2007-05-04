#ifndef TopObjects_TopJetObject_h
#define TopObjects_TopJetObject_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"

using namespace reco;
using namespace std;


class TopJetObject 
{
   
   public:
      TopJetObject();
      TopJetObject(JetType);
      virtual ~TopJetObject();
            
      void    		setRecJet(JetType);
      void    		setLCalJet(TopJet);
      void    		setBCalJet(TopJet);
      void    		setFitJet(TopParticle);
      void    		setBdiscriminant(double);
      
      JetType 		getRecJet() const;
      TopJet 		getLCalJet() const;
      TopJet 		getBCalJet() const;
      TopParticle  	getFitJet() const;
      double  		getBdiscriminant() const;
      
   protected:
      JetType recJet;
      TopJet lCalJet, bCalJet;
      TopParticle fitJet;
      double bdiscr;
      
};

#endif
