import FWCore.ParameterSet.Config as cms

hltLST = cms.EDProducer('LSTProducer@alpaka',
    lstInput = cms.InputTag('hltInputLST'),
    verbose = cms.bool(False),
    ptCut = cms.double(0.8),
    nopLSDupClean = cms.bool(True),
    tcpLSTriplets = cms.bool(True),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)


from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toModify(hltLST, nopLSDupClean = False,
                             tcpLSTriplets = False)
