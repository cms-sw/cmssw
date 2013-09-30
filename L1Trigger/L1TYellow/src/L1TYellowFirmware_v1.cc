
#include "L1Trigger/L1TYellow/src/L1TYellowFirmware.h"

using namespace std;
using namespace l1t;

L1TYellowAlg_v1::L1TYellowAlg_v1(const L1TYellowParams & dbPars) : db(dbPars) {}

L1TYellowAlg_v1::~L1TYellowAlg_v1(){};

void L1TYellowAlg_v1::processEvent(const L1TYellowDigiCollection & input, L1TYellowOutputCollection & out){}
