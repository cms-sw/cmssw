#include "CondFormats/GeometryObjects/interface/PEcalGeometry.h"
#include <iostream>

PEcalGeometry::PEcalGeometry(){}

PEcalGeometry::PEcalGeometry(std::vector<double> tra, std::vector<double> dim, std::vector<uint32_t> ind){
  std::cout<<"I'm building the Ecal DB object"<<std::endl;
  m_translation.insert(m_translation.end(),tra.begin(),tra.end());
  m_dimension.insert(m_dimension.end(),dim.begin(),dim.end());
  m_indexes.insert(m_indexes.end(),ind.begin(),ind.end());
}



