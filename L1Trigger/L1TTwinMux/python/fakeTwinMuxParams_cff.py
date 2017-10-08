import FWCore.ParameterSet.Config as cms

twinmuxParamsSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TTwinMuxParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
#ego
bmbtfParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonBarrelParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

#here
fakeTwinMuxParams = cms.ESProducer('L1TTwinMuxParamsESProducer',
    fwVersion = cms.uint32(1),
    useRpcBxForDtBelowQuality = cms.uint32(4),#correct bx only if dt ts is below
    dphiWindowBxShift         = cms.uint32(9999),
    useOnlyRPC                = cms.bool(False),#use trigger segments (ts)  from rpc hit
    useOnlyDT                 = cms.bool(False),# use ts from dt
    useLowQDT                 = cms.bool(False),
    CorrectDTBxwRPC           = cms.bool(True),#correct bx of dt ts based on rpc
    verbose                   = cms.bool(False),

)
