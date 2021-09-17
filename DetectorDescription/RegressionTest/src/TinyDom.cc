#include "DetectorDescription/RegressionTest/src/TinyDom.h"

using namespace std;

void TinyDomPrettyPrint(ostream& os, const TinyDom& dom) { TinyDomWalker walker(dom); }
