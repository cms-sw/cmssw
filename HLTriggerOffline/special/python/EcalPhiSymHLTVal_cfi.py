# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: EcalPhiSymHLTVal_cfi.py,v 1.3 2009/09/22 13:31:05 beaucero Exp $
#
# \author Stefano Argiro
#
RelValEcalPhiSymMon = cms.EDAnalyzer("HLTAlCaMonEcalPhiSym",
    # product to monitor
    AlCaStreamEBTag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEB"),
    SaveToFile = cms.untracked.bool(False),
    FileName = cms.untracked.string('MonitorAlCaEcalPhiSym.root'),
    AlCaStreamEETag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEE"),
    prescaleFactor = cms.untracked.int32(1),
    # DQM folder to write to
    FolderName = cms.untracked.string('HLT/EcalPhiSym')
)



