import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BPAGTrigRateAnalyzer_cfi import *

JpsiMuMu = bpagTrigOffDQM.clone(
    MassParameters = cms.untracked.vdouble(120, 2.8, 3.4),
    PtParameters = cms.untracked.vdouble(1., 3., 4., 5., 6., 8., 10., 20.)
)
JpsiMuMu.customCollection[0].collectionName = cms.untracked.string ("probeJpsiMuon")

UpsilonMuMu = bpagTrigOffDQM.clone(
    MassParameters = cms.untracked.vdouble(100, 8.5, 10.5),
    PtParameters = cms.untracked.vdouble(1., 3., 4., 5., 6., 8., 10., 20., 50.)
)
UpsilonMuMu.customCollection[0].collectionName = cms.untracked.string ("probeUpsilonMuon")

ZMuMu = bpagTrigOffDQM.clone(
    MassParameters = cms.untracked.vdouble(100, 65, 115),
    PtParameters = cms.untracked.vdouble(10.0, 30.0, 40., 50., 60., 100.0)
)
ZMuMu.customCollection[0].collectionName = cms.untracked.string ("probeZMuon")

TnPEfficiency = cms.Sequence(JpsiMuMu*UpsilonMuMu*ZMuMu)

