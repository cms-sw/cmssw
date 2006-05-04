#ifndef DQMQualityTestsConfiguration_H
#define DQMQualityTestsConfiguration_H


#include<string>

namespace dqm{

  namespace qtest_config{
    
    static std::string type="type";
    
    static std::string  XRangeContent="ContentsXRangeROOT";
    static unsigned int XRangeParams=4;

    static std::string  YRangeContent="ContentsYRangeROOT";
    static unsigned int YRangeParams=4;

    static std::string  DeadChannel="DeadChannelROOT";
    static unsigned int DeadChannelParams=1;

    static std::string  NoisyChannel="NoisyChannelROOT";
    static unsigned int NoisyChannelParams=2;

    static std::string  Comp2RefChi2="Comp2RefChi2";
    static unsigned int Comp2RefChi2Params=4;

    static std::string  Comp2RefKolmogorov="Comp2RefKolmogorov";
    static unsigned int Comp2RefKolmogorovParams=4;

    static std::string  Comp2RefEqualsString="ContentsYRange";
    static unsigned int Comp2RefEqualsStringParams=4;

    static std::string  Comp2RefEqualInt="Comp2RefEqualInt";
    static unsigned int Comp2RefEqualIntParams=4;

    static std::string  Comp2RefEqualFloat="Comp2RefEqualFloat";
    static unsigned int Comp2RefEqualFloatParams=4;

    static std::string  Comp2RefEqualH1="Comp2RefEqualH1";
    static unsigned int Comp2RefEqualH1Params=4;
    
    static std::string  Comp2RefEqualH2="Comp2RefEqualH2";
    static unsigned int Comp2RefEqualH2Params=4;

    static std::string  Comp2RefEqualH3="Comp2RefEqualH3";
    static unsigned int Comp2RefEqualH3Params=4;

  }
  
}

#endif
