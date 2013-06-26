#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

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

  std::cout << "Testing SiStripConfObject" << std::endl;

  SiStripConfObject conf;
  std::cout << "putting par1 with value 1" << std::endl;
  conf.put("par1", 1);
  std::cout << "putting par2 with value 2" << std::endl;
  conf.put("par2", 2);

  std::cout << "Reading back parameters" << std::endl;
  std::cout << "getting par1 = " << conf.get("par1") << std::endl;
  errors += test( conf.get("par1") == 1 );

  std::cout << "getting par2 = " << conf.get("par2") << std::endl;
  errors += test( conf.get("par2") == 2 );

  std::cout << "Trying to read back a non-existent parameter, the test expects an error" << std::endl;
  std::cout << "getting par3 (non-existent) " << conf.get("par3") << std::endl; 
  errors += test( conf.get("par3") == -1 );

  if( errors == 0 ) {
    std::cout << "All tests passed" << std::endl;
  }
  else {
    std::cout << "ERROR: There were " << errors << " tests not passed" << std::endl;
  }

}
