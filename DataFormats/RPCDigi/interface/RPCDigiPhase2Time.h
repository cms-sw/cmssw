#include "DataFormats/RPCDigi/interface/RPCDigiPhase2.h"

class RPCDigiPhase2Time {
public:
  RPCDigiPhase2Time(const RPCDigiPhase2& adigi);
  float time();

private:
  RPCDigiPhase2 theDigi;
};
