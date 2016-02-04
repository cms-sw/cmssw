#include <cassert>
#include <cstddef>
#include <string>
#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace
{
  // Forward declare test helpers
  edm::ModuleDescription makeModuleDescription(std::string const& label);
  void setup_ctx(edm::CurrentProcessingContext& ctx);

  // Icky global junk, to mock lifetime of ModuleDescription.
  static edm::ModuleDescription moduleA = makeModuleDescription("aaa");
  static std::string pathName("path_a");
  static std::size_t pathNumber(21);
  static std::size_t slotInPath(13);  

  static edm::ModuleDescription const* p_moduleA = &moduleA;
  static std::string const*            p_pathName = &pathName;

  // Test helpers
  edm::ModuleDescription makeModuleDescription(std::string const& label)
  {
    edm::ModuleDescription temp("", label);
    return temp;    
  }
  
  void setup_ctx(edm::CurrentProcessingContext& ctx)
  {
    assert(p_moduleA);
    edm::CurrentProcessingContext temp(p_pathName, pathNumber, false);
    temp.activate(slotInPath, p_moduleA);
    ctx = temp;
  }

} // unnamed namespace


void test_default_ctor()
{
  edm::CurrentProcessingContext ctx;
  assert(ctx.moduleLabel() == 0);
  assert(ctx.moduleDescription() == 0);
  assert(ctx.slotInPath() == -1);
  assert(ctx.pathInSchedule() == -1);  
}

void test_activate()
{
  edm::CurrentProcessingContext ctx(p_pathName, pathNumber, false);
  ctx.activate(slotInPath, p_moduleA);
  {
    edm::CurrentProcessingContext const& r_ctx = ctx;
    assert(r_ctx.moduleDescription() == p_moduleA);
    assert(r_ctx.moduleLabel());
    assert(*r_ctx.moduleLabel() == "aaa");
    assert(r_ctx.slotInPath() == 13);
    assert(r_ctx.pathInSchedule() == 21);
  }  
}

void test_deactivate()
{
  edm::CurrentProcessingContext ctx;
  setup_ctx(ctx);
  ctx.deactivate();
  assert(ctx.moduleLabel() == 0);
  assert(ctx.moduleDescription() == 0);  
}


int work()
{
  test_default_ctor();
  test_deactivate();
  return 0;
}

int main()
{
  int rc = -1;
  try { rc = work(); }
  catch (cms::Exception& x) {
      std::cerr << "cms::Exception caught\n";
      std::cerr << x.what() << '\n';
      rc = -2;
  }
  catch (std::exception& x) {
      std::cerr << "std::exception caught\n";
      std::cerr << x.what() << '\n';
      rc = -3;
  }
  catch (...) {
      std::cerr << "Unknown exception caught\n";
      rc = -4;
  }
  return rc;      
}
