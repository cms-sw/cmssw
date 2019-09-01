import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_
DiJetVBFmonitoring = hltobjmonitoring.clone()
#DiJetVBFmonitoring.FolderName = cms.string('HLT/Higgs/VBFMET/DiJet/')
DiJetVBFmonitoring.FolderName = cms.string('HLT/HIG/VBFMET/DiJet/')
DiJetVBFmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults","","HLT" )
DiJetVBFmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiJet110_35_Mjj650_PFMET110_v*","HLT_DiJet110_35_Mjj650_PFMET120_v*","HLT_DiJet110_35_Mjj650_PFMET130_v*")
#DiJetVBFmonitoring.metSelection = cms.string("")
DiJetVBFmonitoring.jetSelection = cms.string("pt>40 & abs(eta)<4.7")
DiJetVBFmonitoring.jetId = cms.string("loose")
DiJetVBFmonitoring.njets = cms.int32(2)
#DiJetVBFmonitoring.metSelection = cms.string("pt>150")

TripleJetVBFmonitoring = DiJetVBFmonitoring.clone()
#TripleJetVBFmonitoring.FolderName = cms.string('HLT/Higgs/VBFMET/TripleJet/')
TripleJetVBFmonitoring.FolderName = cms.string('HLT/HIG/VBFMET/TripleJet/')
TripleJetVBFmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TripleJet110_35_35_Mjj650_PFMET110_v*","HLT_TripleJet110_35_35_Mjj650_PFMET120_v*","HLT_TripleJet110_35_35_Mjj650_PFMET130_v*")

higgsinvHLTJetMETmonitoring = cms.Sequence(
    DiJetVBFmonitoring
    *TripleJetVBFmonitoring
)
