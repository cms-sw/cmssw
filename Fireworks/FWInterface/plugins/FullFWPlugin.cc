#include "Fireworks/FWInterface/interface/FWFFService.h"
#include "Fireworks/FWInterface/interface/FWFFLooper.h"
#include "Fireworks/FWInterface/interface/FWFFHelper.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Framework/interface/LooperFactory.h"

DEFINE_FWK_SERVICE(FWFFService);
DEFINE_FWK_SERVICE(FWFFHelper);
DEFINE_FWK_LOOPER(FWFFLooper);
