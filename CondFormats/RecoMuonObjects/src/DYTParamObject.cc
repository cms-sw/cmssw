#include "CondFormats/RecoMuonObjects/interface/DYTParamObject.h"

double DYTParamObject::parameter(unsigned int iParam) const
{
  
  if (iParam >= paramSize())
    {
      std::cout << "[DYTParamObject::parameter] " 
		<< "Requested parameter is outside size range (" 
		<< paramSize() << ")." << std::endl;
      return 0.;
    }

  return m_params.at(iParam);

}
