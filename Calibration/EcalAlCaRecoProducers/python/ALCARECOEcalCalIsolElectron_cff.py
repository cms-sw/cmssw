import FWCore.ParameterSet.Config as cms

from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *
from Calibration.EcalAlCaRecoProducers.electronIsolationSequence_cff import *
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
from Calibration.EcalAlCaRecoProducers.electronFilter_cfi import *

#
# The current (as of $date$) ALCA stream for single and double electron
# calibration
#

isolElectronewkHLTFilter = hltHighLevel.clone()
isolElectronewkHLTFilter.throw = False

#
# Here we specify trigger names for both 1E31 and 8E29
# Since have throw = false, this setting will work for both menus
#
#
isolElectronewkHLTFilter.HLTPaths = ['HLT_Ele15_SW_L1R', 'HLT_DoubleEle10_SW_L1R', # 1E31
                                     'HLT_Ele10_LW_L1R', 'HLT_DoubleEle5_SW_L1R']  # 8E29

goodElectrons = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("pixelMatchGsfElectrons"),
    cut = cms.string('et > 15')
)

goodElectronFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodElectrons"),
    minNumber = cms.uint32(1)
)

goodElectrons2 = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("pixelMatchGsfElectrons"),
    cut = cms.string('et > 1')
)

goodElectronFilter2 = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodElectrons2"),
    minNumber = cms.uint32(0)
)



superClusterMerger =  cms.EDFilter("EgammaSuperClusterMerger",
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

seqALCARECOEcalCalElectron = cms.Sequence(isolElectronewkHLTFilter*superClusterMerger*superClusterCands*goodSuperClusters*goodSuperClusterFilter*goodSuperClusters2*goodSuperClusterFilter2*goodElectrons*goodElectronFilter*goodElectrons2*goodElectronFilter2*electronFilter*electronIsolationSequence*seqALCARECOEcalCalElectronRECO)

