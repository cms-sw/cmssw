#include "TString.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawCrystal.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawHodo.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawPn.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTower.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawHeader.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTriggerChannel.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawAdc2249.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawScaler.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTdcTriggers.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTdcInfo.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawTpgChannel.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawLaserPulse.h"
#include "IORawData/Ecal2004TBInputService/interface/TRawPattern.h"
#include "IORawData/Ecal2004TBInputService/interface/TRunInfo.h"
#include "IORawData/Ecal2004TBInputService/interface/H4Geom.h"

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TRawHeader+;
#pragma link C++ class TRawCrystal+;
#pragma link C++ class TRawHodo+;
#pragma link C++ class TRawPn+;
#pragma link C++ class TRawTower+;
#pragma link C++ class TRawTriggerChannel+;
#pragma link C++ class TRawAdc2249+;
#pragma link C++ class TRawLaserPulse+;
#pragma link C++ class TRawPattern+;
#pragma link C++ class TRawScaler+;
#pragma link C++ class TRawTdcInfo;
#pragma link C++ class TRawTdcTriggers+;
#pragma link C++ class TRawTpgChannel+;
#pragma link C++ class TRunInfo+;

#pragma link C++ class H4Geom+;

#endif
