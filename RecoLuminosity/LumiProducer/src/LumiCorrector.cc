#include "RecoLuminosity/LumiProducer/interface/LumiCorrector.h"
/**
these are corrections including unit conversion /mb to /ub 
if unit is already /ub, use e-03
**/
LumiCorrector::LumiCorrector(){
  Occ1Norm_=6.36e3; // For 2.76TeV 1.34e3, for HI 2.214e6
  Occ2Norm_=7.97e3;
  ETNorm_=1.59e3;
  PUNorm_=6.37e3;
  Alpha1_=0.063;
  Alpha2_=-0.0037;
  // map doesn't provide any initialization -> do brute force // For HI Afterglow=1.
  AfterglowMap_[213]=0.992; 
  AfterglowMap_[321]=0.990; 
  AfterglowMap_[423]=0.988; 
  AfterglowMap_[597]=0.985; 
  AfterglowMap_[700]=0.984; 
  AfterglowMap_[873]=0.981; 
  AfterglowMap_[1041]=0.979; 
  AfterglowMap_[1179]=0.977; 
  AfterglowMap_[1317]=0.975;
}

void
LumiCorrector::setNormForAlgo(const std::string& algo,float value){
  if( algo=="OCC1" ){Occ1Norm_=value;return;}
  if( algo=="OCC2" ){Occ2Norm_=value;return;}
  if( algo=="ET" ){ETNorm_=value;return;}
  if( algo=="PU" ){PUNorm_=value;return;}
}
void 
LumiCorrector::setCoefficient(const std::string& name,float value){
  if( name=="ALPHA1" ){Alpha1_=value;return;}
  if( name=="ALPHA2" ){Alpha2_=value;return;}
}
float
LumiCorrector::getNormForAlgo(const std::string& algo)const{
  if( algo=="OCC1" ){return Occ1Norm_;}
  if( algo=="OCC2" ){return Occ2Norm_;}
  if( algo=="ET" ){return ETNorm_;}
  if( algo=="PU" ){return PUNorm_;}
  return 1.0;
}
float 
LumiCorrector::getCoefficient(const std::string& name)const{
  if( name=="ALPHA1" ){ return Alpha1_;}
  if( name=="ALPHA2" ){ return Alpha2_;}
  return 0.0;
}
float 
LumiCorrector::AfterglowFactor(int nBXs){
  float Afterglow = 1.;
  for(std::map<int,float>::iterator it = AfterglowMap_.begin(); it != AfterglowMap_.end(); ++it){
    if (nBXs >= it->first){
      Afterglow = it->second;
    }
  }  
  return Afterglow;
}

float 
LumiCorrector::TotalNormOcc1(float TotLumi_noNorm, int nBXs){
  float AvgLumi = (nBXs>0) ? PUNorm_*TotLumi_noNorm/nBXs : 0.;
  return Occ1Norm_*AfterglowFactor(nBXs)/(1 + Alpha1_*AvgLumi + Alpha2_*AvgLumi*AvgLumi);
}
float 
LumiCorrector::TotalNormOcc2(float TotLumi_noNorm, int nBXs){
  return Occ2Norm_;
}
float 
LumiCorrector::TotalNormET(float TotLumi_noNorm, int nBXs){
  return ETNorm_;
}

