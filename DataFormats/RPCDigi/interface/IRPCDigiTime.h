#include "DataFormats/RPCDigi/interface/IRPCDigi.h"

class IRPCDigiTime {
public:
  IRPCDigiTime(const IRPCDigi& adigi);
  float time();
  float coordinateY();
  float timeLR();
  float timeHR();

private:
  IRPCDigi theDigi;
  float TDC2Time(int BX, int SBX, int FT);
};
