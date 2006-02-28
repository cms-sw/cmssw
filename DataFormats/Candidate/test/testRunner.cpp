#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
using namespace CppUnit;

int main()
{
  TextUi::TestRunner runner;
  runner.addTest( TestFactoryRegistry::getRegistry().makeTest() );
  return runner.run( "", false ) ? 0 : 1;
}
