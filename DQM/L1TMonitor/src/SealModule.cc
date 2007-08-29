#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <DQM/L1TMonitor/interface/L1TLTC.h>
DEFINE_ANOTHER_FWK_MODULE(L1TLTC);

#include <DQM/L1TMonitor/interface/L1TFED.h>
DEFINE_ANOTHER_FWK_MODULE(L1TFED);

#include <DQM/L1TMonitor/interface/L1TCSCTPG.h>
DEFINE_ANOTHER_FWK_MODULE(L1TCSCTPG);

#include <DQM/L1TMonitor/interface/L1TCSCTF.h>
DEFINE_ANOTHER_FWK_MODULE(L1TCSCTF);

#include <DQM/L1TMonitor/interface/L1TDTTPG.h>
DEFINE_ANOTHER_FWK_MODULE(L1TDTTPG);

#include <DQM/L1TMonitor/interface/L1TDTTF.h>
DEFINE_ANOTHER_FWK_MODULE(L1TDTTF);

#include <DQM/L1TMonitor/interface/L1TRPCTPG.h>
DEFINE_ANOTHER_FWK_MODULE(L1TRPCTPG);

#include <DQM/L1TMonitor/interface/L1TRPCTF.h>
DEFINE_ANOTHER_FWK_MODULE(L1TRPCTF);

#include <DQM/L1TMonitor/interface/L1TGMT.h>
DEFINE_ANOTHER_FWK_MODULE(L1TGMT);

#include <DQM/L1TMonitor/interface/L1TECALTPG.h>
DEFINE_ANOTHER_FWK_MODULE(L1TECALTPG);

#include <DQM/L1TMonitor/interface/L1THCALTPG.h>
DEFINE_ANOTHER_FWK_MODULE(L1THCALTPG);

#include <DQM/L1TMonitor/interface/L1TGCT.h>
DEFINE_ANOTHER_FWK_MODULE(L1TGCT);

#include <DQM/L1TMonitor/interface/L1TRCT.h>
DEFINE_ANOTHER_FWK_MODULE(L1TRCT);

#include <DQM/L1TMonitor/interface/L1TGT.h>
DEFINE_ANOTHER_FWK_MODULE(L1TGT);

#include <DQM/L1TMonitor/interface/L1TCompare.h>
DEFINE_ANOTHER_FWK_MODULE(L1TCompare);
