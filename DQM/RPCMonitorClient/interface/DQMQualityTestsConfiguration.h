#ifndef DQMQualityTestsConfiguration_H
#define DQMQualityTestsConfiguration_H


#include<string>
#include<vector>

namespace dqm{

  namespace qtest_config{
    
    static std::string type="type";
    
    static std::string  XRangeContent="ContentsXRangeROOT";
    static std::string  YRangeContent="ContentsYRangeROOT";
    static std::string  DeadChannel="DeadChannelROOT";
    static std::string  NoisyChannel="NoisyChannelROOT";
    static std::string  MeanInExpectedValue="MeanWithinExpectedROOT";
   
   // static std::string  Comp2RefChi2="Comp2RefChi2ROOT";    
   // static std::string  Comp2RefKolmogorov="Comp2RefKolmogorov";
   // static std::string  Comp2RefEqualsString="ContentsYRange";
   // static std::string  Comp2RefEqualInt="Comp2RefEqualInt";
   // static std::string  Comp2RefEqualFloat="Comp2RefEqualFloat";
   // static std::string  Comp2RefEqualH1="Comp2RefEqualH1";    
   // static std::string  Comp2RefEqualH2="Comp2RefEqualH2";
   // static std::string  Comp2RefEqualH3="Comp2RefEqualH3";
 
  }
  
}

#endif
