# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: MonitorAlCaEcalPhisym_cfi.py,v 1.4 2009/10/20 14:18:26 argiro Exp $
#
# \author Stefano Argiro
#
EcalPhiSymMonDQM = cms.EDAnalyzer("HLTAlCaMonEcalPhiSym",
    # product to monitor
    AlCaStreamEBTag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEB"),
    SaveToFile = cms.untracked.bool(False),
    FileName = cms.untracked.string('MonitorAlCaEcalPhiSym.root'),
    AlCaStreamEETag = cms.untracked.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEE"),
    prescaleFactor = cms.untracked.int32(1),
    # DQM folder to write to
    FolderName = cms.untracked.string('AlCaReco/EcalPhiSym')
)



