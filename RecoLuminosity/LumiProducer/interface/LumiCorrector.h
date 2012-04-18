#ifndef LUMICORRECTOR_HH
#define LUMICORRECTOR_HH

#include <map>
#include <string>
class LumiCorrector {

 public:
  LumiCorrector();
  ~LumiCorrector(){}
  void setNormForAlgo(const std::string& algo,float value);
  void setCoefficient(const std::string& name,float value);
  float getNormForAlgo(const std::string& algo)const;
  float getCoefficient(const std::string& name)const;
  float AfterglowFactor(int nBXs);
  float TotalNormOcc1(float TotLumi_noNorm, int nBXs);
  float TotalNormOcc2(float TotLumi_noNorm, int nBXs);
  float TotalNormET(float TotLumi_noNorm, int nBXs);
 private:
  float Occ1Norm_;
  float Occ2Norm_;
  float ETNorm_;
  float PUNorm_;
  float Alpha1_;
  float Alpha2_;
  std::map<int,float> AfterglowMap_;
};

#endif
