#include "CondTools/Luminosity/interface/LumiRetrieverBase.h"
edm::ParameterSetID 
lumi::LumiRetrieverBase::parametersetId() const{
  return m_pset.id();
}
