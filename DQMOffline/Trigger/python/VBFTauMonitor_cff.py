import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_
VBFtaumonitoring = hltobjmonitoring.clone()

VBFtaumonitoring.doMETHistos = cms.bool(False)
VBFtaumonitoring.doJetHistos = cms.bool(True)

VBFtaumonitoring.met       = cms.InputTag("pfMet")
VBFtaumonitoring.jets      = cms.InputTag("ak4PFJetsCHS")
VBFtaumonitoring.electrons = cms.InputTag("gedGsfElectrons")
VBFtaumonitoring.muons     = cms.InputTag("muons")

VBFtaumonitoring.FolderName = cms.string('HLT/Higgs/VBFTau')
VBFtaumonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
VBFtaumonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring(
  "HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg_v*",
  "HLT_VBF_DoubleMediumChargedIsoPFTau20_Trk1_eta2p1_Reg_v*",
  "HLT_VBF_DoubleTightChargedIsoPFTau20_Trk1_eta2p1_Reg_v*"
)
VBFtaumonitoring.jetSelection = cms.string("pt>40 & abs(eta)<4.7")
VBFtaumonitoring.jetId = cms.string("loose")
VBFtaumonitoring.njets = cms.int32(2)

higgstautauHLTVBFmonitoring = cms.Sequence(
  VBFtaumonitoring
)
