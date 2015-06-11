import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *

from DQMOffline.JetMET.metDQMConfig_cff     import *
from DQMOffline.JetMET.jetAnalyzer_cff   import *
from DQMOffline.JetMET.SUSYDQMAnalyzer_cfi  import *
from DQMOffline.JetMET.goodOfflinePrimaryVerticesDQM_cfi import *
from RecoJets.JetProducers.PileupJetID_cfi  import *


pileupJetIdProducer.jets = cms.InputTag("ak4PFJets")
pileupJetIdProducer.algos = cms.VPSet(full_5x_chs,cutbased)

pileupJetIdProducerChs.jets = cms.InputTag("ak4PFJetsCHS")

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

dqmAk4PFL1FastL2L3ResidualCorrector = ak4PFL1FastL2L3ResidualCorrector.clone()
dqmAk4PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    #ak4PFL1FastjetCorrector*ak4PFL2RelativeCorrector*ak4PFL3AbsoluteCorrector*ak4PFResidualCorrector*
    dqmAk4PFL1FastL2L3ResidualCorrector
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
dqmCorrPfMetType1=corrPfMetType1.clone()
dqmCorrPfMetType1.jetCorrLabel = cms.InputTag('dqmAk4PFL1FastL2L3Corrector')

jetDQMAnalyzerAk4CaloUncleaned.JetCorrections  = cms.InputTag("dqmAk4CaloL2L3Corrector")
jetDQMAnalyzerAk4CaloCleaned.JetCorrections    = cms.InputTag("dqmAk4CaloL2L3Corrector")
jetDQMAnalyzerAk4PFUncleaned.JetCorrections    = cms.InputTag("dqmAk4PFL1FastL2L3Corrector")
jetDQMAnalyzerAk4PFCleaned.JetCorrections      = cms.InputTag("dqmAk4PFL1FastL2L3Corrector")
jetDQMAnalyzerAk4PFCHSCleaned.JetCorrections   = cms.InputTag("dqmAk4PFCHSL1FastL2L3Corrector")

caloMetDQMAnalyzer.JetCorrections    = cms.InputTag("dqmAk4CaloL2L3Corrector");
pfMetDQMAnalyzer.JetCorrections      = cms.InputTag("dqmAk4PFL1FastL2L3Corrector");
pfMetT1DQMAnalyzer.JetCorrections    = cms.InputTag("dqmAk4PFCHSL1FastL2L3Corrector");

jetMETDQMOfflineSource = cms.Sequence(HBHENoiseFilterResultProducer*goodOfflinePrimaryVerticesDQM*AnalyzeSUSYDQM*pileupJetIdProducer*pileupJetIdProducerChs*#QGTagger*
                                      jetPreDQMSeq*
                                      dqmAk4CaloL2L3CorrectorChain*dqmAk4PFL1FastL2L3CorrectorChain*dqmAk4PFCHSL1FastL2L3CorrectorChain*dqmAk4PFL1FastL2L3ResidualCorrectorChain*
                                      dqmCorrPfMetType1*pfMetT1*
                                      jetDQMAnalyzerSequence*METDQMAnalyzerSequence)

jetMETDQMOfflineSourceMiniAOD = cms.Sequence(goodOfflinePrimaryVerticesDQMforMiniAOD*jetDQMAnalyzerSequenceMiniAOD*METDQMAnalyzerSequenceMiniAOD)
