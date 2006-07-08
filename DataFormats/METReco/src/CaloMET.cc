#include "DataFormats/METReco/interface/CaloMET.h"

using namespace reco;

bool CaloMET::overlap( const Candidate & ) const 
{
  return false;
}
