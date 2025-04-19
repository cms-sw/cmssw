import FWCore.ParameterSet.Config as cms

hltLST = cms.EDProducer('LSTProducer@alpaka',
    lstInput = cms.InputTag('hltInputLST'),
    verbose = cms.bool(False),
    ptCut = cms.double(0.8),
    nopLSDupClean = cms.bool(False),
    tcpLSTriplets = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
(seedingLST & trackingLST).toModify(hltLST, nopLSDupClean = True,
                                            tcpLSTriplets = True )
