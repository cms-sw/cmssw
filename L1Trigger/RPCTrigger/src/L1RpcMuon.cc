//---------------------------------------------------------------------------

#include "L1Trigger/RPCTrigger/src/L1RpcMuon.h"
#include <sstream>
#include <iomanip>
//---------------------------------------------------------------------------
using namespace std;

string L1RpcMuon::ToString() const {
  std::ostringstream ostr;
  ostr<<"Muon: PtCode: "<<setw(2)<<PtCode<<", Quality: "<<Quality<<", Sign: "<<Sign
      <<", Tower: "<<setw(2)<<ConeCrdnts.Tower<<", LogSector: "<<setw(2)<<ConeCrdnts.LogSector
			<<", LogSegment "<<setw(2)<<ConeCrdnts.LogSegment;
    
  ostr<<", firedPl: "<<bitset<6>(FiredPlanes);

  ostr<<", PattNum: "<<PatternNum<<" RefStrip: "<<RefStripNum<<endl;
  return ostr.str();
}
