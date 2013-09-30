
#include "L1Trigger/L1TYellow/interface/L1TYellowAlg.h"
#include "L1TYellowFirmware.h"

using namespace std;

namespace l1t {
  L1TYellowAlg * NewL1TYellowAlg(const L1TYellowParams & dbPars){
    return new L1TYellowAlg_v1(dbPars);
  }
}
