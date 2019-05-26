#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "ClusterFilterFactory.h"

#include "ClusterFilterByAlgo.h"
#include "ClusterFilterByAlgoOrSize.h"
#include "ClusterFilterBySize.h"

using namespace ticl;

DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgo, "ClusterFilterByAlgo");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterByAlgoOrSize, "ClusterFilterByAlgoOrSize");
DEFINE_EDM_PLUGIN(ClusterFilterFactory, ClusterFilterBySize, "ClusterFilterBySize");
