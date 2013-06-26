#include "DetectorDescription/Core/interface/DDConstant.h"
#include <iostream>

using namespace std;

int main () {
  DDName n("martin");
  cout << n << endl;
  vector<double> vals(10,3.141);
  DDConstant c(n,&vals);
  cout << c.value() << endl;
  cout << c << endl;
  DDConstant c2("liendl",&vals);
  cout << c.name() << endl;
  cout << c2.name() << endl;
  cout << c2 << endl;
  DDConstant c3(n);
  cout << c3 << endl;
  return 0;
}
