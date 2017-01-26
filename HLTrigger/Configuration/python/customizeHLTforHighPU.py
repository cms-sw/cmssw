import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *

# CMSSW version specific customizations
def customizeHLTforHighPU(process):
    for module in producers_by_type(process,"SiPixelClusterProducer"):
        if hasattr(module, "maxNumberOfClusters"):
            module.maxNumberOfClusters = cms.int32( 80000 ) # default: 20000

    for module in producers_by_type(process,"SeedGeneratorFromRegionHitsEDProducer"):
        if hasattr(module,"OrderedHitsFactoryPSet"):
            OrderedHitsFactory_pset = module.OrderedHitsFactoryPSet
            if hasattr(OrderedHitsFactory_pset,"GeneratorPSet"):
                Generator_pset = OrderedHitsFactory_pset.GeneratorPSet
                if hasattr(Generator_pset,"maxElement"):
#                    if getattr(Generator_pset,"maxElement") != 100000:
#                        print module, Generator_pset.maxElement
                    Generator_pset.maxElement = 100000 # default 100000 for pp modules, 1000000 for PA modules
    
        if hasattr(module,"ClusterCheckPSet"):
            ClusterCheck_pset = module.ClusterCheckPSet
            if hasattr(ClusterCheck_pset,"MaxNumberOfCosmicClusters"):
#                if getattr(ClusterCheck_pset,"MaxNumberOfCosmicClusters") != 800000:
#                    print module,ClusterCheck_pset.MaxNumberOfCosmicClusters
                ClusterCheck_pset.MaxNumberOfCosmicClusters = 800000 # default 50000 for pp modules, 400000 for PA modules
            if hasattr(ClusterCheck_pset,"MaxNumberOfPixelClusters"):
#                if getattr(ClusterCheck_pset,"MaxNumberOfPixelClusters") != 80000:
#                    print module,ClusterCheck_pset.MaxNumberOfPixelClusters
                ClusterCheck_pset.MaxNumberOfPixelClusters = 80000 # default 10000 for pp modules, 40000 for PA modules 
    return process
