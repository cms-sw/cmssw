#include "CondFormats/RecoMuonObjects/interface/DYTParamObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

double DYTParamObject::parameter(unsigned int iParam) const
{
  
  if (iParam >= paramSize())
    {
      edm::LogWarning("DYTParamObject") 
	<< "The requested parameter (" << (iParam + 1) 
	<< ") is outside size range (" << paramSize() << ").";
      return 0.;
    }

  return m_params.at(iParam);

}
