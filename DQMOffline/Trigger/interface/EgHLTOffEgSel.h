#ifndef  DQMOFFLINE_TRIGGER_EGHLTOFFEGSEL
#define  DQMOFFLINE_TRIGGER_EGHLTOFFEGSEL

//this class works out which cuts the electron/photon passes/fails
//why am I rolling my own, simply put there is no electron/photon cut class that I know off
//which will allow isolation and id variables to be cut on at the same time and return
//a int with the bits corresponding to cut pass/fail
//also I'm going to need to modify this to keep up with trigger cuts
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgHLTEgCutValues.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

namespace edm{
  class ParameterSet;
}

namespace egHLT {
  class OffEle;
  class OffPho;

  class OffEgSel  {
    
 private:

    EgCutValues ebCutValues_;
    EgCutValues eeCutValues_;
    
  public:
    OffEgSel(){}//default, it doesnt to anything
    explicit OffEgSel(const edm::ParameterSet& config){setup(config);}
    ~OffEgSel(){} //we own nothing so default destructor, copy and assignment okay
    
     
    bool passCuts(const OffEle& ele,int cutMask=~0x0)const{return getCutCode(ele,cutMask)==0x0;}
    int getCutCode(const OffEle& ele,int cutMask=~0x0)const;  
    static int getCutCode(const OffEle& ele,const EgCutValues& cuts,int cutMask=~0x0);
    
    bool passCuts(const OffPho& pho,int cutMask=~0x0)const{return getCutCode(pho,cutMask)==0x0;}
    int getCutCode(const OffPho& pho,int cutMask=~0x0)const;
    static int getCutCode(const OffPho& pho,const EgCutValues& cuts,int cutMask=~0x0);

    void setEBCuts(const EgCutValues& cuts){ebCutValues_=cuts;}
    void setEECuts(const EgCutValues& cuts){eeCutValues_=cuts;}
    
    const EgCutValues& ebCuts()const{return ebCutValues_;}
    const EgCutValues& eeCuts()const{return eeCutValues_;}
    
    void setup(const edm::ParameterSet&);
    
    
  };
}

#endif
