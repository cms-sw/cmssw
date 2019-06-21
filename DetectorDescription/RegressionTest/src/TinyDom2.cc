#include "DetectorDescription/RegressionTest/src/TinyDom2.h"

using namespace std;

void TinyDom2PrettyPrint(ostream& os, const TinyDom2& dom) { TinyDom2Walker walker(dom); }
