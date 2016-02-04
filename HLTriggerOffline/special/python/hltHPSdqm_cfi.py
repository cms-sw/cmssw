import FWCore.ParameterSet.Config as cms

hltHPSdqm = cms.EDAnalyzer('DQMHcalPhiSymHLT',
folderName=cms.string("HLT/AlCa_HcalPhiSym"),
SaveToRootFile=cms.bool(False),
outputRootFileName=cms.string("hltHPSdqm.root"),
rawInputLabel=cms.InputTag("hltAlCaHcalFEDSelector")
)
