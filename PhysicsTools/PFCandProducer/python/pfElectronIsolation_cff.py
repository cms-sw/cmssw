import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaSuperClusterMerger_cfi import *
from RecoEgamma.EgammaIsolationAlgos.egammaBasicClusterMerger_cfi import *


pfeleIsoDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
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




pfEleIsoDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfeleIsoDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01', 
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfeleIsoChDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfElectronsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        inputCandView = cms.InputTag("pfAllChargedHadrons"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )

)

pfeleIsoNeDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfElectronsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        inputCandView = cms.InputTag("pfAllNeutralHadrons"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )

)

pfeleIsoGaDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfElectronsPtGt5"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        inputCandView = cms.InputTag("pfAllPhotons"),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string('')
    )

)




pfEleIsoChDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfeleIsoChDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfEleIsoNeDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfeleIsoNeDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfEleIsoGaDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfeleIsoGaDepositPFCandidates"),
        deltaR = cms.double(0.5),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(2.0)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)







pfElectrons  = cms.EDFilter("IsolatedPFCandidateSelector",
    src = cms.InputTag("pfElectronsPtGt5"),
    IsoDeposit = cms.InputTag("pfEleIsoDeposit"),
    IsolationCut = cms.double(2.5),       
    ) 



pfElectronIsol = cms.Sequence(    
  pfeleIsoDepositPFCandidates
  *pfEleIsoDeposit
  *pfeleIsoChDepositPFCandidates
  *pfEleIsoChDeposit
  *pfeleIsoNeDepositPFCandidates
  *pfEleIsoNeDeposit
  *pfeleIsoGaDepositPFCandidates
  *pfEleIsoGaDeposit
  *pfElectrons
  )
