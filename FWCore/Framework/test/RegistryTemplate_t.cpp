#include <cassert>
#include "FWCore/ParameterSet/interface/Registry.h"

int main()
{
  edm::pset::Registry* psreg = 
    edm::pset::Registry::instance();

  assert( psreg );
}
