# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: EcalPhiSymHLTVal_cfi.py,v 1.4 2009/10/19 18:17:54 beaucero Exp $
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



