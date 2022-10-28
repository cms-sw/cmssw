import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_
VBFtaumonitoring = hltobjmonitoring.clone(

    #FolderName = 'HLT/Higgs/VBFTau',
    FolderName = 'HLT/HIG/VBFTau',
    numGenericTriggerEventPSet = dict(hltInputTag   = "TriggerResults::HLT",
                                    hltPaths = ["HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg_v*",
                                                "HLT_VBF_DoubleMediumChargedIsoPFTau20_Trk1_eta2p1_Reg_v*",
                                                "HLT_VBF_DoubleTightChargedIsoPFTau20_Trk1_eta2p1_Reg_v*"]),
    jetSelection = "pt>40 & abs(eta)<4.7",
    jetId = "loose",
    njets = 2,

    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg_v*"])
)
higgstautauHLTVBFmonitoring = cms.Sequence(
  VBFtaumonitoring
)
