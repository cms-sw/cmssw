#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include <iostream>

using namespace reco;

void PreId::setMatching(MatchingType type,bool result,unsigned n)
{
  if(n<matching_.size())
    {
      if(result)
	{
	  matching_[n] |= (1 << type);
	}
      else
	{
	  matching_[n] &= ~(1 <<type);
	}
    }
  else
    {
      std::cout << " Out of range " << std::endl;
    }
}

float PreId::mva(unsigned n) const
{
  if(n<mva_.size())
    return mva_[n];
  return -999.;
}
