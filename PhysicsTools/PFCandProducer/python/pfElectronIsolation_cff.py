import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaSuperClusterMerger_cfi import *
from RecoEgamma.EgammaIsolationAlgos.egammaBasicClusterMerger_cfi import *


pfEleIsoDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfElectronsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        inputCandView = cms.InputTag("particleFlow"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )
                                             
)




pfEleIsoDeposit = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfEleIsoDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01', 
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfEleIsoChDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfElectronsPtGt5"),
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

pfEleIsoNeDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfElectronsPtGt5"),
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

pfEleIsoGaDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfElectronsPtGt5"),
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


pfEleIsoChDeposit = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfEleIsoChDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfEleIsoNeDeposit = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfEleIsoNeDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfEleIsoGaDeposit = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfEleIsoGaDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)







pfElectrons  = cms.EDProducer("IsolatedPFCandidateSelector",
    src = cms.InputTag("pfElectronsPtGt5"),
    IsoDeposit = cms.InputTag("pfEleIsoDeposit"),
    IsolationCut = cms.double(2.5),       
    ) 



pfElectronIsolationSequence = cms.Sequence(    
  pfEleIsoDepositPFCandidates
  + pfEleIsoDeposit
  + pfEleIsoChDepositPFCandidates
  + pfEleIsoChDeposit
  + pfEleIsoNeDepositPFCandidates
  + pfEleIsoNeDeposit
  + pfEleIsoGaDepositPFCandidates
  + pfEleIsoGaDeposit
  + pfElectrons
  )
