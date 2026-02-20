import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *

hltBackend = cms.EDProducer( "AlpakaBackendProducer@alpaka"
)

hltStatusOnGPUFilter = cms.EDFilter( "AlpakaBackendFilter",
    producer = cms.InputTag( 'hltBackend','backend' ),
    backends = cms.vstring( 'CudaAsync','ROCmAsync' )
)

DQM_TRKHeterogeneousValidation = cms.Path(
    HLTBeginSequence
    + hltBackend
    + hltStatusOnGPUFilter
    + HLTTrackingSequence
    + HLTTrackingSequenceSerialSync
)
