# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: MonitorAlCaEcalPhisym.cfi,v 1.1 2008/06/18 15:16:10 argiro Exp $
#
# \author Stefano Argiro
#
EcalPhiSymMon = cms.EDFilter("DQMSourcePhiSym",
    # product to monitor
    AlCaStreamEBTag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEB"),
    SaveToFile = cms.untracked.bool(False),
    FileName = cms.untracked.string('MonitorAlCaEcalPhiSym.root'),
    AlCaStreamEETag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEE"),
    prescaleFactor = cms.untracked.int32(1),
    # DQM folder to write to
    FolderName = cms.untracked.string('ALCAStreamEcalPhiSym')
)



