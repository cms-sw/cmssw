#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"


#include <DQM/L1TMonitor/interface/L1TFED.h>
DEFINE_FWK_MODULE(L1TFED);

#include <DQM/L1TMonitor/interface/L1TCSCTPG.h>
DEFINE_FWK_MODULE(L1TCSCTPG);

#include <DQM/L1TMonitor/interface/L1TCSCTF.h>
DEFINE_FWK_MODULE(L1TCSCTF);

#include <DQM/L1TMonitor/interface/L1TDTTPG.h>
DEFINE_FWK_MODULE(L1TDTTPG);

#include <DQM/L1TMonitor/interface/L1TDTTF.h>
DEFINE_FWK_MODULE(L1TDTTF);

#include <DQM/L1TMonitor/interface/L1TRPCTPG.h>
DEFINE_FWK_MODULE(L1TRPCTPG);

#include <DQM/L1TMonitor/interface/L1TRPCTF.h>
DEFINE_FWK_MODULE(L1TRPCTF);

#include <DQM/L1TMonitor/interface/L1TGMT.h>
DEFINE_FWK_MODULE(L1TGMT);


#include <DQM/L1TMonitor/interface/L1TGCT.h>
DEFINE_FWK_MODULE(L1TGCT);

#include <DQM/L1TMonitor/interface/L1TRCT.h>
DEFINE_FWK_MODULE(L1TRCT);

#include "DQM/L1TMonitor/interface/L1TPUM.h"
DEFINE_FWK_MODULE(L1TPUM);

#include <DQM/L1TMonitor/interface/L1TGT.h>
DEFINE_FWK_MODULE(L1TGT);

#include <DQM/L1TMonitor/interface/L1TCompare.h>
DEFINE_FWK_MODULE(L1TCompare);

#include "DQM/L1TMonitor/interface/BxTiming.h"
DEFINE_FWK_MODULE(BxTiming);

//Emulator DQM:

#include "DQM/L1TMonitor/interface/L1TDEMON.h"
DEFINE_FWK_MODULE(L1TDEMON);

#include "DQM/L1TMonitor/interface/L1TdeGCT.h"
DEFINE_FWK_MODULE(L1TdeGCT);

#include "DQM/L1TMonitor/interface/L1TdeRCT.h"
DEFINE_FWK_MODULE(L1TdeRCT);

#include "DQM/L1TMonitor/interface/L1TdeCSCTF.h"
DEFINE_FWK_MODULE(L1TdeCSCTF);

#include "DQM/L1TMonitor/interface/L1THIonImp.h"
DEFINE_FWK_MODULE(L1THIonImp);

//#include "DQM/L1TMonitor/interface/L1GtHwValidation.h"
//DEFINE_FWK_MODULE(L1GtHwValidation);
