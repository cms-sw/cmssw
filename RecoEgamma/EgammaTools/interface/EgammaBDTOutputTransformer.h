#ifndef RecoEgamma_ElectronTools_EgammaBDTOutputTransformer_h
#define RecoEgamma_ElectronTools_EgammaBDTOutputTransformer_h

//author: Sam Harper (RAL)
//description: 
//  translates the raw value returned by the BDT output to the true value
//  apparently its MINUIT-like, orginally taken from E/gamma regression applicator

#include <vdt/vdtMath.h>

class EgammaBDTOutputTransformer {

public:  
  EgammaBDTOutputTransformer(const double limitLow,const double limitHigh):
    offset_(limitLow + 0.5*(limitHigh-limitLow)),
    scale_(0.5*(limitHigh-limitLow)){}

  double operator()(const double rawVal)const{return offset_ + scale_*vdt::fast_sin(rawVal);}
 
private:
  const double offset_;
  const double scale_;
};


#endif
