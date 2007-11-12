#ifndef DQMQualityTestsConfiguration_H
#define DQMQualityTestsConfiguration_H


#include<string>
#include<vector>

namespace dqm{

  namespace qtest_config{
    
    static std::string type="type";
    
    static std::string  XRangeContent       = "ContentsXRangeROOT";
    static std::string  YRangeContent       = "ContentsYRangeROOT";
    static std::string  DeadChannel         = "DeadChannelROOT";
    static std::string  NoisyChannel        = "NoisyChannelROOT";
    static std::string  MeanInExpectedValue = "MeanWithinExpectedROOT";
    static std::string  MostProbableLandau  = "MostProbableLandauROOT";
 
  }
  
}

#endif
