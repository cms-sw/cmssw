#include <cassert>
#include "DataFormats/Common/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"

int main()
{
  edm::ModuleDescriptionRegistry* mreg = 
    edm::ModuleDescriptionRegistry::instance();

  edm::ProcessHistoryRegistry* pnlreg = 
    edm::ProcessHistoryRegistry::instance();

  assert( mreg );
  assert( pnlreg );
}
