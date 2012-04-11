#include "RecoLuminosity/LumiProducer/interface/LumiCorrector.h"

LumiCorrector::LumiCorrector(){
  Norm_=7.13;
  PUNorm_=6.37;
  Alpha1_=0.063;
  Alpha2_=-0.0037;
  // map doesn't provide any initialization -> do brute force
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
LumiCorrector::LumiCorrector(float norm,float punorm,float alpha1,float alpha2):Norm_(7.13),PUNorm_(6.37),Alpha1_(0.063),Alpha2_(-0.0037){
  // map doesn't provide any initialization -> do brute force
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
LumiCorrector::TotalCorrectionFactor(float TotLumi_noNorm, int nBXs){
  if(nBXs==0) return 1.0;
  float AvgLumi = PUNorm_*TotLumi_noNorm/nBXs;
  return Norm_*AfterglowFactor(nBXs)/(1 + Alpha1_*AvgLumi + Alpha2_*AvgLumi*AvgLumi);
}
