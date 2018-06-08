#include "RecoEgamma/EgammaIsolationAlgos/interface/EGHcalRecHitSelector.h"

EGHcalRecHitSelector::EGHcalRecHitSelector(const edm::ParameterSet& config):
  maxDIEta_(config.getParameter<int>("maxDIEta")),
  maxDIPhi_(config.getParameter<int>("maxDIPhi")),
  minEnergyHCAL_(config.getParameter<double>("minEnergyHCAL"))
{

}

edm::ParameterSetDescription EGHcalRecHitSelector::makePSetDescription()
{
 edm::ParameterSetDescription desc;
 desc.add<int>("maxDIEta",5);
 desc.add<int>("maxDIPhi",5);
 desc.add<double>("minEnergyHCAL",0.8); 
 return desc; 
}

int EGHcalRecHitSelector::calDIEta(int iEta1,int iEta2)
{  
  int dEta = iEta1-iEta2;
  if(iEta1*iEta2<0) {//-ve to +ve transistion and no crystal at zero
    if(dEta<0) dEta++;
    else dEta--;
  }
  return dEta;
}


int EGHcalRecHitSelector::calDIPhi(int iPhi1,int iPhi2)
{
  int dPhi = iPhi1-iPhi2;
  if(dPhi>72/2) dPhi-=72;
  else if(dPhi<-72/2) dPhi+=72;
  return dPhi;
}
