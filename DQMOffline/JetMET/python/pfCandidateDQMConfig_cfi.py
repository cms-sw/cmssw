import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETDQMCleanup_cff import *
from DQMOffline.JetMET.metDiagnosticParameterSet_cfi import *
from DQMOffline.JetMET.metDiagnosticParameterSetMiniAOD_cfi import *

#jet corrector defined in jetMETDQMOfflineSource python file

pfCandidateDQMAnalyzer = cms.EDAnalyzer("DQMPFCandidateAnalyzer",  
    CandType=cms.untracked.string('PFCand'),
    PFCandidateLabel = cms.InputTag('particleFlow', ''),

    ptMinCand      = cms.double(1.),
    hcalMin      =cms.double(1.),  

    CleaningParameters = cleaningParameters.clone(       
        bypassAllPVChecks = cms.bool(False)
        ),
    METDiagonisticsParameters = multPhiCorr_METDiagnostics,

    FilterResultsLabelMiniAOD  = cms.InputTag("TriggerResults::RECO"),
    FilterResultsLabelMiniAOD2  = cms.InputTag("TriggerResults::reRECO"), 

    LSBegin = cms.int32(0),
    LSEnd   = cms.int32(-1),      

 
    HBHENoiseLabelMiniAOD = cms.string("Flag_HBHENoiseFilter"),
    HBHENoiseFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResult"),
    HBHENoiseIsoFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHEIsoNoiseFilterResult"),

    verbose     = cms.int32(0),

    DCSFilter = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        #DebugOn = cms.untracked.bool(True),
        Filter = cms.untracked.bool(True)
        ),
)

packedCandidateDQMAnalyzerMiniAOD = pfCandidateDQMAnalyzer.clone(
    CandType=cms.untracked.string('Packed'),
    PFCandidateLabel = cms.InputTag('packedPFCandidates', ''),
    METDiagonisticsParameters = multPhiCorr_METDiagnosticsMiniAOD,
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    = cms.InputTag( "goodOfflinePrimaryVerticesDQMforMiniAOD" ),
        ),
)

