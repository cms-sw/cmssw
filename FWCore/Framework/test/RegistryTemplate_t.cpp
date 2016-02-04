#include <cassert>
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/ParameterSet/interface/Registry.h"

int main()
{
  edm::pset::Registry* psreg = 
    edm::pset::Registry::instance();

  edm::ProcessHistoryRegistry* pnlreg = 
    edm::ProcessHistoryRegistry::instance();

  assert( psreg );
  assert( pnlreg );
}
