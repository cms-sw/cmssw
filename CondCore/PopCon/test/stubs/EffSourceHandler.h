#ifndef POPCON_EX_EffSource_H
#define POPCON_EX_EffSource_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"



#include "CondFormats/Calibration/interface/Efficiency.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<string>

namespace popcon{
  class ExEffSource : public popcon::PopConSourceHandler<condex::Efficiency>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~ExEffSource(); 
    ExEffSource(const edm::ParameterSet& pset); 
    
  private:
    std::string m_name;
    long long m_since;
    std::string m_type;
    std::vector<double> m_params;
  };
}
#endif // POPCON_EX_PEDESTALS_SRC_H
