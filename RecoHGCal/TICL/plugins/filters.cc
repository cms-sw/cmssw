#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoHGCal/TICL/interface/ClusterFilterFactory.h"

#include "RecoHGCal/TICL/interface/ClusterFilterByAlgo.h"
#include "RecoHGCal/TICL/interface/ClusterFilterByAlgoOrSize.h"
#include "RecoHGCal/TICL/interface/ClusterFilterBySize.h"

DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgo, "ClusterFilterByAlgo");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgoOrSize, "ClusterFilterByAlgoOrSize");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterBySize, "ClusterFilterBySize");
