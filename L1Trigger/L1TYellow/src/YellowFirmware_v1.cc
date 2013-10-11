
#include "L1Trigger/L1TYellow/src/YellowFirmware.h"

using namespace std;
using namespace l1t;

YellowAlg_v1::YellowAlg_v1(const YellowParams & dbPars) : db(dbPars) {}

YellowAlg_v1::~YellowAlg_v1(){};

void YellowAlg_v1::processEvent(const YellowDigiCollection & input, YellowOutputCollection & out){}
