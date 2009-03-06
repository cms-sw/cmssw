import FWCore.ParameterSet.Config as cms
#from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import *
#from RecoEgamma.EgammaIsolationAlgos.egammaSuperClusterMerger_cfi import *
#from RecoEgamma.EgammaIsolationAlgos.egammaBasicClusterMerger_cfi import *

from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *

pfmuIsoDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
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
        src = cms.InputTag("pfmuIsoDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),#2.0
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfmuIsoChDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfMuonsPtGt5"),
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

pfmuIsoNeDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfMuonsPtGt5"),
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

pfmuIsoGaDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfMuonsPtGt5"),
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




pfMuIsoChDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfmuIsoChDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(0.5)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfMuIsoNeDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfmuIsoNeDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(0.5)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

pfMuIsoGaDeposit = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("pfmuIsoGaDepositPFCandidates"),
        deltaR = cms.double(0.3),
        weight = cms.string('1'),
        vetos = cms.vstring('0.01',
            'Threshold(0.5)'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)







pfMuons  = cms.EDFilter("IsolatedPFCandidateSelector",
    src = cms.InputTag("pfMuonsPtGt5"),
    IsoDeposit = cms.InputTag("pfMuIsoDeposit"),
    IsolationCut = cms.double(2.5),#2.5       
    ) 



pfMuonIsol = cms.Sequence(    
  pfmuIsoDepositPFCandidates
  *pfMuIsoDeposit
  *pfmuIsoChDepositPFCandidates
  *pfMuIsoChDeposit
  *pfmuIsoNeDepositPFCandidates
  *pfMuIsoNeDeposit
  *pfmuIsoGaDepositPFCandidates
  *pfMuIsoGaDeposit
  *pfMuons
  )
