import FWCore.ParameterSet.Config as cms

#
# $Id: EcalPhiSymHLTVal_cfi.py,v 1.2 2009/01/20 09:56:41 nuno Exp $
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
