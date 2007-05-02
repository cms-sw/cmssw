#ifndef TopObjects_TopMuonObject_h
#define TopObjects_TopMuonObject_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopObject.h"

using namespace reco;
using namespace std;


class TopMuonObject 
{
   
   public:
      TopMuonObject();
      TopMuonObject(TopMuon);
      virtual ~TopMuonObject();
            
      void    		setRecMuon(TopMuon);
      void    		setFitMuon(TopParticle);
      void    		setLRvalue(double);
      
      TopMuon 		getRecMuon() const;
      TopParticle  	getFitMuon() const;
      double 		getLRvalue() const;
      
   protected:
      TopMuon 		recMuon;
      TopParticle 	fitMuon;
      double 		LRvalue;
      
};

#endif
