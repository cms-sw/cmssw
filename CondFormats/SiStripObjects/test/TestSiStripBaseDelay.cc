#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"

#include <iostream>

int test( const bool testResult )
{
  if( testResult ) {
    return 0;
  }
  std::cout << "test not passed" << std::endl;
  return 1;
}

int main()
{
  int errors = 0;

  std::cout << "Testing SiStripBaseDelay" << std::endl;

  SiStripBaseDelay delay;
  std::cout << "Storing delay values from local file" << std::endl;
  delay.put(1, 0, 0);

  std::cout << "Reading back parameters" << std::endl;
  errors += test( delay.coarseDelay(1) == 0 );

  if( errors == 0 ) {
    std::cout << "All tests passed" << std::endl;
  }
  else {
    std::cout << "ERROR: There were " << errors << " tests not passed" << std::endl;
  }

}
