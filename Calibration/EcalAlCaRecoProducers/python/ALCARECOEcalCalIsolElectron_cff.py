import FWCore.ParameterSet.Config as cms

from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *
from Calibration.EcalAlCaRecoProducers.electronIsolationSequence_cff import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

import HLTrigger.HLTfilters.hltHighLevel_cfi 

#
# The current (as of $Date: 2010/03/04 05:42:57 $) ALCA stream for single and double electron
# calibration
#

isolElectronewkHLTFilter=HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#   HLTPaths = ['HLT_Ele15_SW_L1R', 'HLT_DoubleEle10_SW_L1R', # 1E31
#               'HLT_Ele10_LW_L1R', 'HLT_DoubleEle5_SW_L1R']  # 8E29
    eventSetupPathsKey='EcalCalElectron',
    throw = False
)

goodElectrons = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string('et > 15')
)

goodElectronFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodElectrons"),
    minNumber = cms.uint32(1)
)

goodElectrons2 = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string('et > 1')
)

goodElectronFilter2 = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodElectrons2"),
    minNumber = cms.uint32(0)
)



superClusterMerger =  cms.EDProducer("EgammaSuperClusterMerger",
    src = cms.VInputTag(cms.InputTag('correctedHybridSuperClusters'), cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'))
    )

superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
    src = cms.InputTag("superClusterMerger"),
    particleType = cms.string('gamma')
)

goodSuperClusters = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("superClusterCands"),
    cut = cms.string('et > 20')
)

goodSuperClusterFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodSuperClusters"),
    minNumber = cms.uint32(1)
)

goodSuperClusters2 = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("superClusterCands"),
    cut = cms.string('et > 10')
)

goodSuperClusterFilter2 = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodSuperClusters2"),
    minNumber = cms.uint32(0)
)

seqALCARECOEcalCalElectronRECO = cms.Sequence(alCaIsolatedElectrons)

seqALCARECOEcalCalElectron = cms.Sequence(isolElectronewkHLTFilter*         \
                                          superClusterMerger*               \
                                          superClusterCands*                \
                                          goodSuperClusters*                \
                                          goodSuperClusterFilter*           \
                                          goodSuperClusters2*               \
                                          goodSuperClusterFilter2*          \
                                          goodElectrons*                    \
                                          goodElectronFilter*               \
                                          goodElectrons2*                   \
                                          goodElectronFilter2*              \
                                          alcaElectronIsolationSequence*    \
                                          seqALCARECOEcalCalElectronRECO)

