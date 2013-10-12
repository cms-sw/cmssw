
#include "YellowFirmwareImp.h"

using namespace std;
using namespace l1t;

YellowFirmwareImp2::YellowFirmwareImp2(const YellowParams & dbPars) : db(dbPars) {}

YellowFirmwareImp2::~YellowFirmwareImp2(){};

void YellowFirmwareImp2::processEvent(const YellowDigiCollection & input, YellowOutputCollection & out){}
