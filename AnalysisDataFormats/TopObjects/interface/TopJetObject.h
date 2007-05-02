#ifndef TopObjects_TopJetObject_h
#define TopObjects_TopJetObject_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopObject.h"

using namespace reco;
using namespace std;


class TopJetObject 
{
   
   public:
      TopJetObject();
      TopJetObject(jetType);
      virtual ~TopJetObject();
            
      void    		setRecJet(jetType);
      void    		setLCalJet(TopJet);
      void    		setBCalJet(TopJet);
      void    		setFitJet(TopParticle);
      void    		setBdiscriminant(double);
      
      jetType 		getRecJet() const;
      TopJet 		getLCalJet() const;
      TopJet 		getBCalJet() const;
      TopParticle  	getFitJet() const;
      double  		getBdiscriminant() const;
      
   protected:
      jetType recJet;
      TopJet lCalJet, bCalJet;
      TopParticle fitJet;
      double bdiscr;
      
};

#endif
