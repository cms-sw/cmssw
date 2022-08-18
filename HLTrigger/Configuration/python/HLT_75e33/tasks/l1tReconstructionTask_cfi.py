import FWCore.ParameterSet.Config as cms

from ..modules.L1EGammaClusterEmuProducer_cfi import *
from ..modules.l1EGammaEEProducer_cfi import *
from ..modules.l1NNTauProducerPuppi_cfi import *

from ..modules.simCaloStage2Layer1Digis_cfi import *
from ..modules.simCscTriggerPrimitiveDigis_cfi import *
from ..modules.simDtTriggerPrimitiveDigis_cfi import *
from ..modules.simEmtfDigis_cfi import *
from ..modules.simGmtCaloSumDigis_cfi import *
from ..modules.simGmtStage2Digis_cfi import *
from ..modules.simKBmtfDigis_cfi import *
from ..modules.simKBmtfStubs_cfi import *
from ..modules.simMuonGEMPadDigiClusters_cfi import *
from ..modules.simMuonGEMPadDigis_cfi import *
from ..modules.simOmtfDigis_cfi import *
from ..modules.simTwinMuxDigis_cfi import *

l1tReconstructionTask = cms.Task(
    L1EGammaClusterEmuProducer,
    l1EGammaEEProducer,
    l1NNTauProducerPuppi,
    simCaloStage2Layer1Digis,
    simCscTriggerPrimitiveDigis,
    simDtTriggerPrimitiveDigis,
    simEmtfDigis,
    simGmtCaloSumDigis,
    simGmtStage2Digis,
    simKBmtfDigis,
    simKBmtfStubs,
    simMuonGEMPadDigiClusters,
    simMuonGEMPadDigis,
    simOmtfDigis,
    simTwinMuxDigis
)
