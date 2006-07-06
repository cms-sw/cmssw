#include <cassert>
#include "FWCore/Framework/interface/ModuleDescriptionRegistry.h"
#include "FWCore/Framework/interface/ProcessHistoryRegistry.h"

int main()
{
  edm::ModuleDescriptionRegistry* mreg = 
    edm::ModuleDescriptionRegistry::instance();

  edm::ProcessHistoryRegistry* pnlreg = 
    edm::ProcessHistoryRegistry::instance();

  assert( mreg );
  assert( pnlreg );
}
