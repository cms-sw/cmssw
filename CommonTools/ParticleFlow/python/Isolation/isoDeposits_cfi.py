import FWCore.ParameterSet.Config as cms

# template isodeposit producer
# for now, used both for electrons and musons, but can be specialised
isoDeposits = cms.EDProducer(
    "CandIsoDepositProducer",
    src = cms.InputTag(""),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
    Diff_z = cms.double(99999.99), #(0.2)
    ComponentName = cms.string('CandViewExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(99999.99), #(0.1)
    inputCandView = cms.InputTag(""),
    DR_Veto = cms.double(1e-05),
    DepositLabel = cms.untracked.string('')
    )
    )
