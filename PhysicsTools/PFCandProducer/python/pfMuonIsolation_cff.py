import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *

pfMuIsoDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfMuonsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),#(0.2)
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),#(0.1)
        inputCandView = cms.InputTag("particleFlow"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )                                            
)

pfMuIsoDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfMuIsoDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),#2.0
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfMuIsoChDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfMuonsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        inputCandView = cms.InputTag("allChargedHadrons"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )

)

pfMuIsoNeDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfMuonsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        inputCandView = cms.InputTag("allNeutralHadrons"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )

)

pfMuIsoGaDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfMuonsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        inputCandView = cms.InputTag("allPhotons"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )

)




pfMuIsoChDeposit = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfMuIsoChDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(0.5)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfMuIsoNeDeposit = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfMuIsoNeDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(0.5)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfMuIsoGaDeposit = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfMuIsoGaDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(0.5)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)


pfMuons  = cms.EDProducer("IsolatedPFCandidateSelector",
    src = cms.InputTag("pfMuonsPtGt5"),
    IsoDeposit = cms.InputTag("pfMuIsoDeposit"),
    IsolationCut = cms.double(2.5),#2.5       
    ) 


pfMuonIsolationSequence = cms.Sequence(    
  pfMuIsoDepositPFCandidates
  + pfMuIsoDeposit
  + pfMuIsoChDepositPFCandidates
  + pfMuIsoChDeposit
  + pfMuIsoNeDepositPFCandidates
  + pfMuIsoNeDeposit
  + pfMuIsoGaDepositPFCandidates
  + pfMuIsoGaDeposit
  + pfMuons
  )
