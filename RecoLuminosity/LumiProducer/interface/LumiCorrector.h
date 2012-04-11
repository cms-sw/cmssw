#ifndef LUMICORRECTOR_HH
#define LUMICORRECTOR_HH

#include <map>

class LumiCorrector {

  public:
    LumiCorrector();
    LumiCorrector(float norm=7.13,float punorm=6.37,float alpha1=0.063,float alpha2=-0.0037);
    ~LumiCorrector(){}

    float AfterglowFactor(int nBXs);
    float TotalCorrectionFactor(float TotLumi_noNorm, int nBXs);

    float Norm_;
    float PUNorm_;
    float Alpha1_;
    float Alpha2_;
    std::map<int,float> AfterglowMap_;
};

#endif
