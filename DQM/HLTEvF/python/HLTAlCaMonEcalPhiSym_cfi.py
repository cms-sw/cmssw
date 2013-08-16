import FWCore.ParameterSet.Config as cms

#
#
# \author Stefano Argiro
#
EcalPhiSymMon = cms.EDAnalyzer("HLTAlCaMonEcalPhiSym",
    # product to monitor
    AlCaStreamEBTag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEB"),
    SaveToFile = cms.untracked.bool(False),
    FileName = cms.untracked.string('MonitorAlCaEcalPhiSym.root'),
    AlCaStreamEETag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEE"),
    prescaleFactor = cms.untracked.int32(1),
    # DQM folder to write to
    FolderName = cms.untracked.string('HLT/EcalPhiSym')
)
