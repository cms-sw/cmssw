#ifndef TopObjects_TopMETObject_h
#define TopObjects_TopMETObject_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopObject.h"

using namespace reco;
using namespace std;


class TopMETObject 
{
   
   public:
      TopMETObject();
      TopMETObject(TopMET);
      virtual ~TopMETObject();
            
      void    		setRecMET(TopMET);
      void    		setFitMET(TopParticle);
      
      TopMET 		getRecMET() const;
      TopParticle  	getFitMET() const;
      
   protected:
      TopMET       recMET;
      TopParticle  fitMET;
      
};

#endif
