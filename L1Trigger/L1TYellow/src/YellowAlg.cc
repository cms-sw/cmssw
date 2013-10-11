#include "L1Trigger/L1TYellow/interface/YellowAlg.h"
#include "YellowFirmware.h"

using namespace std;

namespace l1t {

  YellowAlg * NewYellowAlg(const YellowParams & dbPars){
    cout << "Using firmware version:  " << dbPars.firmwareVersion() << "\n";

    return new YellowAlg_v1(dbPars);
  }

}
