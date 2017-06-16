import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_
EXO_VBF_METmonitoring = hltobjmonitoring.clone()
EXO_VBF_METmonitoring.FolderName = cms.string('HLT/Higgs/VBFMET/EXOVBF/')
EXO_VBF_METmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults","","reHLT" )
EXO_VBF_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_EXO_VBF")
#EXO_VBF_METmonitoring.metSelection = cms.string("")
#EXO_VBF_METmonitoring.jetSelection = cms.string("pt>40 & (eta<4.7 | eta>-4.7)")
#EXO_VBF_METmonitoring.njets = cms.int32(2)
#EXO_VBF_METmonitoring.metSelection = cms.string("pt>150")

EXO_VBF_3_METmonitoring = hltobjmonitoring.clone()
EXO_VBF_3_METmonitoring.FolderName = cms.string('HLT/Higgs/VBFMET/EXOVBF3/')
EXO_VBF_3_METmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults","","reHLT" )
EXO_VBF_3_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_EXO_VBF_3")
#EXO_VBF_3_METmonitoring.jetSelection = cms.string("pt>40 & fabs(eta)<4.7")
#EXO_VBF_3_METmonitoring.njets = cms.int32(2)

higgsinvHLTJetMETmonitoring = cms.Sequence(
    EXO_VBF_METmonitoring
    *EXO_VBF_3_METmonitoring
)

