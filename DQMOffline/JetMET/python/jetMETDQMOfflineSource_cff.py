import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.metDQMConfig_cff     import *
from DQMOffline.JetMET.jetAnalyzer_cff   import *
from DQMOffline.JetMET.pfCandidateDQMConfig_cfi   import *
from DQMOffline.JetMET.SUSYDQMAnalyzer_cfi  import *
from DQMOffline.JetMET.goodOfflinePrimaryVerticesDQM_cfi import *
from RecoJets.JetProducers.PileupJetID_cfi  import *
from RecoJets.JetProducers.QGTagger_cfi  import *
from RecoMET.METFilters.metFilters_cff import*

HcalStripHaloFilterDQM=HcalStripHaloFilter.clone(
    taggingMode = cms.bool(True))
CSCTightHaloFilterDQM=CSCTightHaloFilter.clone(
    taggingMode = cms.bool(True))
CSCTightHalo2015FilterDQM=CSCTightHalo2015Filter.clone(
    taggingMode = cms.bool(True))
eeBadScFilterDQM=eeBadScFilter.clone(
    taggingMode = cms.bool(True))
EcalDeadCellTriggerPrimitiveFilterDQM=EcalDeadCellTriggerPrimitiveFilter.clone(
    taggingMode = cms.bool(True))
EcalDeadCellBoundaryEnergyFilterDQM=EcalDeadCellBoundaryEnergyFilter.clone(
    taggingMode = cms.bool(True)) 

pileupJetIdCalculatorDQM=pileupJetIdCalculator.clone(
    jets = cms.InputTag("ak4PFJets"),
    jec = cms.string("AK4PF"),
    applyJec = cms.bool(True),
    inputIsCorrected = cms.bool(False)
)

pileupJetIdEvaluatorDQM=pileupJetIdEvaluator.clone(
    jets = cms.InputTag("ak4PFJets"),
    jetids = cms.InputTag("pileupJetIdCalculatorDQM"),
    jec = cms.string("AK4PF"),
    applyJec = cms.bool(True),
    inputIsCorrected = cms.bool(False)
)

pileupJetIdCalculatorCHSDQM=pileupJetIdCalculator.clone(
    applyJec = cms.bool(True),
    inputIsCorrected = cms.bool(False),
)

pileupJetIdEvaluatorCHSDQM=pileupJetIdEvaluator.clone(
    jetids = cms.InputTag("pileupJetIdCalculatorCHSDQM"),
    applyJec = cms.bool(True),
    inputIsCorrected = cms.bool(False)
    )

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4CaloL2L3ResidualCorrectorChain,ak4CaloL2L3ResidualCorrector,ak4CaloResidualCorrector,ak4CaloL2L3Corrector,ak4CaloL3AbsoluteCorrector,ak4CaloL2RelativeCorrector

dqmAk4CaloL2L3ResidualCorrector = ak4CaloL2L3ResidualCorrector.clone()
dqmAk4CaloL2L3ResidualCorrectorChain = cms.Sequence(
    #ak4CaloL2RelativeCorrector*ak4CaloL3AbsoluteCorrector*ak4CaloResidualCorrector*
    dqmAk4CaloL2L3ResidualCorrector
)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFL1FastL2L3ResidualCorrectorChain,ak4PFL1FastL2L3ResidualCorrector,ak4PFCHSL1FastL2L3Corrector,ak4PFResidualCorrector,ak4PFL3AbsoluteCorrector,ak4PFL2RelativeCorrector,ak4PFL1FastjetCorrector

dqmAk4PFCHSL1FastL2L3Corrector = ak4PFCHSL1FastL2L3Corrector.clone()
dqmAk4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(
    #ak4CaloL2RelativeCorrector*ak4CaloL3AbsoluteCorrector*ak4CaloResidualCorrector*
    dqmAk4PFCHSL1FastL2L3Corrector
)

dqmAk4PFL1FastL2L3ResidualCorrector = ak4PFL1FastL2L3ResidualCorrector.clone()
dqmAk4PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    #ak4PFL1FastjetCorrector*ak4PFL2RelativeCorrector*ak4PFL3AbsoluteCorrector*ak4PFResidualCorrector*
    dqmAk4PFL1FastL2L3ResidualCorrector
)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSL1FastL2L3ResidualCorrectorChain,ak4PFCHSL1FastL2L3ResidualCorrector,ak4PFCHSL1FastL2L3Corrector,ak4PFCHSResidualCorrector,ak4PFCHSL3AbsoluteCorrector,ak4PFCHSL2RelativeCorrector,ak4PFCHSL1FastjetCorrector

dqmAk4PFCHSL1FastL2L3ResidualCorrector = ak4PFCHSL1FastL2L3ResidualCorrector.clone()
dqmAk4PFCHSL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    #ak4PFCHSL1FastjetCorrector*ak4PFCHSL2RelativeCorrector*ak4PFCHSL3AbsoluteCorrector*ak4PFCHSResidualCorrector
    dqmAk4PFCHSL1FastL2L3ResidualCorrector
)

dqmAk4PFCHSL1FastL2L3Corrector = ak4PFCHSL1FastL2L3Corrector.clone()
dqmAk4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(
    dqmAk4PFCHSL1FastL2L3Corrector
)

HBHENoiseFilterResultProducerDQM=HBHENoiseFilterResultProducer.clone()

jetPreDQMSeq=cms.Sequence(ak4CaloL2RelativeCorrector*ak4CaloL3AbsoluteCorrector*ak4CaloResidualCorrector*
                          ak4PFL1FastjetCorrector*ak4PFL2RelativeCorrector*ak4PFL3AbsoluteCorrector*ak4PFResidualCorrector*
                          ak4PFCHSL1FastjetCorrector*ak4PFCHSL2RelativeCorrector*ak4PFCHSL3AbsoluteCorrector*ak4PFCHSResidualCorrector)

from JetMETCorrections.Type1MET.correctedMet_cff import pfMetT1
from JetMETCorrections.Type1MET.correctionTermsPfMetType0PFCandidate_cff import *
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import corrPfMetType1

dqmCorrPfMetType1=corrPfMetType1.clone(jetCorrLabel = cms.InputTag('dqmAk4PFCHSL1FastL2L3Corrector'),
                                       jetCorrLabelRes = cms.InputTag('dqmAk4PFCHSL1FastL2L3ResidualCorrector')
                                       )
pfMETT1=pfMetT1.clone(srcCorrections = cms.VInputTag(
        cms.InputTag('dqmCorrPfMetType1', 'type1')
        ))

jetMETDQMOfflineSource = cms.Sequence(AnalyzeSUSYDQM*QGTagger*
                                      pileupJetIdCalculatorCHSDQM*pileupJetIdEvaluatorCHSDQM*
                                      pileupJetIdCalculatorDQM*pileupJetIdEvaluatorDQM*
                                      jetPreDQMSeq*
                                      dqmAk4CaloL2L3ResidualCorrectorChain*dqmAk4PFL1FastL2L3ResidualCorrectorChain*dqmAk4PFCHSL1FastL2L3ResidualCorrectorChain*dqmAk4PFCHSL1FastL2L3CorrectorChain*
                                      goodOfflinePrimaryVerticesDQM*                                                                            
                                      dqmCorrPfMetType1*pfMETT1*jetDQMAnalyzerSequence*HBHENoiseFilterResultProducer*
                                      CSCTightHaloFilterDQM*CSCTightHalo2015FilterDQM*eeBadScFilterDQM*EcalDeadCellTriggerPrimitiveFilterDQM*EcalDeadCellBoundaryEnergyFilterDQM*HcalStripHaloFilterDQM                                      
                                      *METDQMAnalyzerSequence
                                      *pfCandidateDQMAnalyzer)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

jetMETDQMOfflineRedoProductsMiniAOD = cms.Sequence(goodOfflinePrimaryVerticesDQMforMiniAOD)

jetMETDQMOfflineSourceMiniAOD = cms.Sequence(jetDQMAnalyzerSequenceMiniAOD*METDQMAnalyzerSequenceMiniAOD*packedCandidateDQMAnalyzerMiniAOD)
