#ifndef POPCON_EX_PEDESTALS_SRC_H
#define POPCON_EX_PEDESTALS_SRC_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"



#include "CondFormats/Calibration/interface/Pedestals.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include<string>

namespace popcon{
  class ExPedestalSource : public popcon::PopConSourceHandler<Pedestals>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~ExPedestalSource(); 
    ExPedestalSource(const edm::ParameterSet& pset); 
    
  private:
    std::string m_name;
    long long m_since;
    long long m_increment;
    long long m_number;
  };
}
#endif // POPCON_EX_PEDESTALS_SRC_H
