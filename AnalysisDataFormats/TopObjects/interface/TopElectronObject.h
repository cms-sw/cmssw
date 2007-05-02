#ifndef TopObjects_TopElectronObject_h
#define TopObjects_TopElectronObject_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopObject.h"

using namespace reco;
using namespace std;


class TopElectronObject 
{
   
   public:
      TopElectronObject();
      TopElectronObject(TopElectron);
      virtual ~TopElectronObject();
            
      void    		setRecElectron(TopElectron);
      void    		setFitElectron(TopParticle);
      void    		setLRvalue(double);
      
      TopElectron 	getRecElectron() const;
      TopParticle  	getFitElectron() const;
      double 		getLRvalue() const;
      
   protected:
      TopElectron 	recElectron;
      TopParticle 	fitElectron;
      double 		LRvalue;
      
};

#endif
