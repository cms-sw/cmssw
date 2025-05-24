#include "DataFormats/RPCDigi/interface/RPCDigiPhase2Time.h"

RPCDigiPhase2Time::RPCDigiPhase2Time(const RPCDigiPhase2& adigi) : theDigi(adigi) {}

float RPCDigiPhase2Time::time() {
  return 25. * theDigi.bx() + 1.5625 * theDigi.sbx();  // 25./16. = 1.5625 ns
}
