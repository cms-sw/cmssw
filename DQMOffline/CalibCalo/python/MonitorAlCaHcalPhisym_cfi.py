# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: MonitorAlCaEcalPhisym_cfi.py,v 1.2 2008/08/14 09:33:36 argiro Exp $
#
# \author Stefano Argiro
#
HcalPhiSymMon = cms.EDAnalyzer("DQMHcalPhiSymAlCaReco",
    # product to monitor
    hbheInputMB = cms.InputTag("hbherecoMB"),
    hoInputMB = cms.InputTag("horecoMB"),
    hfInputMB = cms.InputTag("hfrecoMB"),
    hbheInputNoise = cms.InputTag("hbherecoNoise"),
    hoInputNoise = cms.InputTag("horecoNoise"),
    hfInputNoise = cms.InputTag("hfrecoNoise"),
    # File to save 
    SaveToFile = cms.untracked.bool(True),
    FileName = cms.untracked.string('MonitorAlCaHcalPhiSym.root'),
    # DQM folder to write to
    FolderName = cms.untracked.string('ALCAStreamHcalPhiSym')
)



