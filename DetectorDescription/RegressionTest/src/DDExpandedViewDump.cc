namespace std { } using namespace std;
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <set>
#include <algorithm>
#include <string>
#include "DetectorDescription/RegressionTest/interface/DDExpandedViewDump.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"

void DDExpandedViewDump(ostream & os, DDExpandedView & ex, size_t skip, size_t sto)
{
  typedef set<string>::iterator s_iter;
  set<string> result;
  bool go(true);
  int  count(0);
  bool dotrans(true);
  if (getenv("DDNOTRANS")) dotrans=false;
  ++skip;
  while(go) {
    if ((count % 5000)==0) cout << count << ' ' << flush;
    if (sto > 0) if ((count % sto)==0) break;
    ++count;
    if((count % skip) ==0){
      stringstream s;
      s.setf(ios_base::fixed,ios_base::floatfield);
      s.precision(2);
      s << ex.logicalPart().name() << ' '
	<< ex.copyno() << ' ' 
	<< ex.geoHistory() << " r="
	<< ex.geoHistory().back().posdata()->rot_.name() << "\n"; 
      DDRotationMatrix rm = ex.rotation();
      s << "R=(" << rm.xx() << ' ' << rm.xy() << ' ' << rm.xz() << endl
        << "   " << rm.yx() << ' ' << rm.yy() << ' ' << rm.yz() << endl
        << "   " << rm.zx() << ' ' << rm.zy() << ' ' << rm.zz() << endl;
      rm = ex.geoHistory().back().posdata()->rotation();
      s << "r=(" << rm.xx() << ' ' << rm.xy() << ' ' << rm.xz() << endl
        << "   " << rm.yx() << ' ' << rm.yy() << ' ' << rm.yz() << endl
        << "   " << rm.zx() << ' ' << rm.zy() << ' ' << rm.zz() << endl;
      if (dotrans) {
      s << "T=("
	<< ex.translation().x() << ','
	<< ex.translation().y() << ','
	<< ex.translation().z() << ") "
	;
      }      
      pair<s_iter,bool> ins = result.insert(s.str());
      if(!ins.second) {
	cout << "DDExpandedViewDump:ERROR: duplicated=" << s.str() << endl;
      } 
    }
    go = ex.next();
  }
  s_iter it(result.begin()), ed(result.end());
  for(;it != ed;++it) {
    os << *it << endl;
  }
}
