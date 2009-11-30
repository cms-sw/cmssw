#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

#include <iostream>

using namespace std;

int test( const bool testResult )
{
  if( testResult ) {
    return 0;
  }
  cout << "test not passed" << endl;
  return 1;
}

int main()
{
  int errors = 0;

  cout << "Testing SiStripConfObject" << endl;

  SiStripConfObject conf;
  cout << "putting par1 with value 1" << endl;
  conf.put("par1", 1);
  cout << "putting par2 with value 2" << endl;
  conf.put("par2", 2);

  cout << "Reading back parameters" << endl;
  cout << "getting par1 = " << conf.get("par1") << endl;
  errors += test( conf.get("par1") == 1 );

  cout << "getting par2 = " << conf.get("par2") << endl;
  errors += test( conf.get("par2") == 2 );

  cout << "Trying to read back a non-existent parameter, the test expects an error" << endl;
  cout << "getting par3 (non-existent) " << conf.get("par3") << endl; 
  errors += test( conf.get("par3") == -1 );

  if( errors == 0 ) {
    cout << "All tests passed" << endl;
  }
  else {
    cout << "ERROR: There were " << errors << " tests not passed" << endl;
  }

}
