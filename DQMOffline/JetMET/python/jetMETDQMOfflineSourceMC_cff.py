import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *

from DQMOffline.JetMET.metDQMConfig_cff     import *
from DQMOffline.JetMET.jetAnalyzer_cff   import *
from DQMOffline.JetMET.SUSYDQMAnalyzer_cfi  import *
from DQMOffline.JetMET.goodOfflinePrimaryVerticesDQM_cfi import *
from RecoJets.JetProducers.PileupJetID_cfi  import *
from RecoJets.JetProducers.QGTagger_cfi  import *

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

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4CaloL2L3CorrectorChain,ak4CaloResidualCorrector,ak4CaloL2L3Corrector,ak4CaloL3AbsoluteCorrector,ak4CaloL2RelativeCorrector

dqmAk4CaloL2L3Corrector = ak4CaloL2L3Corrector.clone()
dqmAk4CaloL2L3CorrectorChain = cms.Sequence(
    #ak4CaloL2RelativeCorrector*ak4CaloL3AbsoluteCorrector*
    dqmAk4CaloL2L3Corrector
)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFL1FastL2L3CorrectorChain,ak4PFL1FastL2L3Corrector,ak4PFL3AbsoluteCorrector,ak4PFL2RelativeCorrector,ak4PFL1FastjetCorrector

dqmAk4PFL1FastL2L3Corrector = ak4PFL1FastL2L3Corrector.clone()
dqmAk4PFL1FastL2L3CorrectorChain = cms.Sequence(
    #ak4PFL1FastjetCorrector*ak4PFL2RelativeCorrector*ak4PFL3AbsoluteCorrector*
    dqmAk4PFL1FastL2L3Corrector
)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSL1FastL2L3CorrectorChain,ak4PFCHSL1FastL2L3Corrector,ak4PFCHSL3AbsoluteCorrector,ak4PFCHSL2RelativeCorrector,ak4PFCHSL1FastjetCorrector

dqmAk4PFCHSL1FastL2L3Corrector = ak4PFCHSL1FastL2L3Corrector.clone()
dqmAk4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(
    #ak4PFCHSL1FastjetCorrector*ak4PFCHSL2RelativeCorrector*ak4PFCHSL3AbsoluteCorrector
    dqmAk4PFCHSL1FastL2L3Corrector
)

jetPreDQMSeq=cms.Sequence(ak4CaloL2RelativeCorrector*ak4CaloL3AbsoluteCorrector*
                          ak4PFL1FastjetCorrector*ak4PFL2RelativeCorrector*ak4PFL3AbsoluteCorrector*
                          ak4PFCHSL1FastjetCorrector*ak4PFCHSL2RelativeCorrector*ak4PFCHSL3AbsoluteCorrector)

from JetMETCorrections.Type1MET.correctedMet_cff import pfMetT1
from JetMETCorrections.Type1MET.correctionTermsPfMetType0PFCandidate_cff import *
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import corrPfMetType1

dqmCorrPfMetType1=corrPfMetType1.clone(jetCorrLabel = cms.InputTag('dqmAk4PFCHSL1FastL2L3Corrector'),
                                       jetCorrLabelRes = cms.InputTag('dqmAk4PFCHSL1FastL2L3Corrector')
                                       )
pfMETT1=pfMetT1.clone(srcCorrections = cms.VInputTag(
        cms.InputTag('dqmCorrPfMetType1', 'type1')
    ))

jetDQMAnalyzerAk4CaloUncleanedMC=jetDQMAnalyzerAk4CaloUncleaned.clone(JetCorrections  = cms.InputTag("dqmAk4CaloL2L3Corrector"))
jetDQMAnalyzerAk4CaloCleanedMC=jetDQMAnalyzerAk4CaloCleaned.clone(JetCorrections    = cms.InputTag("dqmAk4CaloL2L3Corrector"))
jetDQMAnalyzerAk4PFUncleanedMC=jetDQMAnalyzerAk4PFUncleaned.clone(JetCorrections    = cms.InputTag("dqmAk4PFL1FastL2L3Corrector"))
jetDQMAnalyzerAk4PFCleanedMC=jetDQMAnalyzerAk4PFCleaned.clone(JetCorrections      = cms.InputTag("dqmAk4PFL1FastL2L3Corrector"))
jetDQMAnalyzerAk4PFCHSCleanedMC=jetDQMAnalyzerAk4PFCHSCleaned.clone(JetCorrections   = cms.InputTag("dqmAk4PFCHSL1FastL2L3Corrector"))

caloMetDQMAnalyzerMC=caloMetDQMAnalyzer.clone(JetCorrections    = cms.InputTag("dqmAk4CaloL2L3Corrector"))
pfMetDQMAnalyzerMC=pfMetDQMAnalyzer.clone(JetCorrections      = cms.InputTag("dqmAk4PFL1FastL2L3Corrector"))
pfMetT1DQMAnalyzerMC=pfMetT1DQMAnalyzer.clone(JetCorrections    = cms.InputTag("dqmAk4PFCHSL1FastL2L3Corrector"))

jetMETDQMOfflineSource = cms.Sequence(HBHENoiseFilterResultProducer*goodOfflinePrimaryVerticesDQM*AnalyzeSUSYDQM*QGTagger*
                                      pileupJetIdCalculatorCHSDQM*pileupJetIdEvaluatorCHSDQM*
                                      pileupJetIdCalculatorDQM*pileupJetIdEvaluatorDQM*
                                      jetPreDQMSeq*
                                      dqmAk4CaloL2L3CorrectorChain*dqmAk4PFL1FastL2L3CorrectorChain*dqmAk4PFCHSL1FastL2L3CorrectorChain*
                                      dqmCorrPfMetType1*pfMETT1*
                                      jetDQMAnalyzerAk4CaloCleanedMC*jetDQMAnalyzerAk4PFUncleanedMC*jetDQMAnalyzerAk4PFCleanedMC*jetDQMAnalyzerAk4PFCHSCleanedMC*
                                      caloMetDQMAnalyzerMC*pfMetDQMAnalyzerMC*pfMetT1DQMAnalyzerMC)

jetMETDQMOfflineSourceMiniAOD = cms.Sequence(goodOfflinePrimaryVerticesDQMforMiniAOD*jetDQMAnalyzerSequenceMiniAOD*METDQMAnalyzerSequenceMiniAOD)
