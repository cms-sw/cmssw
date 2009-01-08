import FWCore.ParameterSet.Config as cms

#from EgammaAnalysis.EgammaIsolationProducers.egammaElectronTkIsolation_cfi import *
#from EgammaAnalysis.EgammaIsolationProducers.egammaHcalIsolation_cfi import *
#from EgammaAnalysis.EgammaIsolationProducers.egammaHOE_cfi import *
#from EgammaAnalysis.EgammaIsolationProducers.egammaEcalIsolationSequence_cff import *

from RecoEgamma.EgammaIsolationAlgos.eleIsolationSequence_cff import *


#egammaElectronSqPtTkIsolation = cms.EDProducer("ElectronSqPtTkIsolationProducer",
#    absolut = cms.bool(False),
#    trackProducer = cms.InputTag("generalTracks"),
#    intRadius = cms.double(0.02),
#    electronProducer = cms.InputTag("electronFilter"),
#    extRadius = cms.double(0.6),
#    ptMin = cms.double(1.5),
#    maxVtxDist = cms.double(0.1)
#)

#electronIsolationSequence = cms.Sequence(egammaElectronSqPtTkIsolation+egammaHcalIsolation+egammaHOE+egammaElectronTkIsolation+egammaEcalIsolationSequence)
# egammaElectronTkIsolation.electronProducer = 'electronFilter'
# egammaElectronTkIsolation.trackProducer = 'generalTracks'
# egammaElectronTkIsolation.ptMin = 1.5
# egammaElectronTkIsolation.intRadius = 0.02
# egammaElectronTkIsolation.extRadius = 0.2
# egammaHcalIsolation.emObjectProducer = 'electronFilter'
# egammaHcalIsolation.etMin = 0.
# egammaHcalIsolation.intRadius = 0.15
# egammaHcalIsolation.extRadius = .3
# egammaHcalIsolation.absolut = False
# egammaHOE.emObjectProducer = 'electronFilter'
# egammaEcalIsolation.emObjectProducer = 'electronFilter'
# egammaEcalIsolation.absolut = False
# egammaBasicClusterMerger.src = cms.VInputTag(cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"), cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"))
# egammaSuperClusterMerger.src =cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"), cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"))


electronIsolationSequence=cms.Sequence(eleIsolationSequence)
