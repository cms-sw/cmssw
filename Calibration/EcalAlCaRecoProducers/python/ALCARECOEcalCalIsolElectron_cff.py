import FWCore.ParameterSet.Config as cms

from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *
from Calibration.EcalAlCaRecoProducers.electronIsolationSequence_cff import *
from Calibration.EcalAlCaRecoProducers.ewkHLTFilter_cfi import *
from Calibration.EcalAlCaRecoProducers.electronFilter_cfi import *

ewkHLTFilter.HLTPaths = ['HLT_LooseIsoEle15_LW_L1R']

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

#testSelector = cms.EDFilter("AssociatedVariableMaxCutCandViewSelector",
#    filter = cms.bool(True),
#    src = cms.InputTag("egammaElectronTkIsolation"),
#    max = cms.double(0.1)
#)

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
#seqALCARECOEcalCalElectron = cms.Sequence(ewkHLTFilter*superClusterMerger*superClusterCands*goodSuperClusters*goodSuperClusterFilter*goodSuperClusters2*goodSuperClusterFilter2*goodElectrons*goodElectronFilter*goodElectrons2*goodElectronFilter2*electronFilter*electronIsolationSequence*testSelector*seqALCARECOEcalCalElectronRECO)
seqALCARECOEcalCalElectron = cms.Sequence(ewkHLTFilter*superClusterMerger*superClusterCands*goodSuperClusters*goodSuperClusterFilter*goodSuperClusters2*goodSuperClusterFilter2*goodElectrons*goodElectronFilter*goodElectrons2*goodElectronFilter2*electronFilter*electronIsolationSequence*seqALCARECOEcalCalElectronRECO)

