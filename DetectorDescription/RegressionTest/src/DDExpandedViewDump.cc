#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDTransform.h"

using namespace std;

void DDExpandedViewDump(ostream& os, DDExpandedView& ex, size_t skip, size_t sto) {
  typedef set<string>::iterator s_iter;
  set<string> result;
  bool go(true);
  int count(0);
  bool dotrans(true);
  if (std::getenv("DDNOTRANS"))
    dotrans = false;
  ++skip;
  while (go) {
    if ((count % 5000) == 0)
      cout << count << ' ' << flush;
    if (sto > 0)
      if ((count % sto) == 0)
        break;
    ++count;
    if ((count % skip) == 0) {
      stringstream s;
      s.setf(ios_base::fixed, ios_base::floatfield);
      s.precision(2);
      s << ex.logicalPart().name() << ' ' << ex.copyno() << ' ' << ex.geoHistory()
        << " r=" << ex.geoHistory().back().posdata()->ddrot().name() << "\n";
      DDRotationMatrix rm = ex.rotation();
      {
        double v[9];
        rm.GetComponents(v, v + 9);
        s << "R=(";
        s << v[0] << ' ' << v[1] << ' ' << v[2] << endl;
        s << v[3] << ' ' << v[4] << ' ' << v[5] << endl;
        s << v[6] << ' ' << v[7] << ' ' << v[7] << endl;
      }
      rm = ex.geoHistory().back().posdata()->rotation();
      {
        double v[9];
        rm.GetComponents(v, v + 9);
        s << "r=(";
        s << v[0] << ' ' << v[1] << ' ' << v[2] << endl;
        s << v[3] << ' ' << v[4] << ' ' << v[5] << endl;
        s << v[6] << ' ' << v[7] << ' ' << v[7] << endl;
      }
      if (dotrans) {
        s << "T=(" << ex.translation().x() << ',' << ex.translation().y() << ',' << ex.translation().z() << ") ";
      }
      pair<s_iter, bool> ins = result.insert(s.str());
      if (!ins.second) {
        cout << "DDExpandedViewDump:ERROR: duplicated=" << s.str() << endl;
      }
    }
    go = ex.next();
  }
  s_iter it(result.begin()), ed(result.end());
  for (; it != ed; ++it) {
    os << *it << endl;
  }
}
