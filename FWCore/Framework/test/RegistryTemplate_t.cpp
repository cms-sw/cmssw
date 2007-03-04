#include <cassert>
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

int main()
{
  edm::ModuleDescriptionRegistry* mreg = 
    edm::ModuleDescriptionRegistry::instance();

  edm::ProcessHistoryRegistry* pnlreg = 
    edm::ProcessHistoryRegistry::instance();

  assert( mreg );
  assert( pnlreg );
}
