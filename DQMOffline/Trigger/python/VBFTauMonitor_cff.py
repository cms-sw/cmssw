import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_
VBFtaumonitoring = hltobjmonitoring.clone()

#VBFtaumonitoring.FolderName = cms.string('HLT/Higgs/VBFTau')
VBFtaumonitoring.FolderName = cms.string('HLT/HIG/VBFTau')
VBFtaumonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults","","HLT" )
VBFtaumonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring(
  "HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg_v*",
  "HLT_VBF_DoubleMediumChargedIsoPFTau20_Trk1_eta2p1_Reg_v*",
  "HLT_VBF_DoubleTightChargedIsoPFTau20_Trk1_eta2p1_Reg_v*"
)
VBFtaumonitoring.jetSelection = cms.string("pt>40 & abs(eta)<4.7")
VBFtaumonitoring.jetId = cms.string("loose")
VBFtaumonitoring.njets = cms.int32(2)

VBFtaumonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring(
  "HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg_v*"
)

higgstautauHLTVBFmonitoring = cms.Sequence(
  VBFtaumonitoring
)
