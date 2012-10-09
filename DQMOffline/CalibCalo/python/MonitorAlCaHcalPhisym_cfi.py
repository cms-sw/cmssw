# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: MonitorAlCaHcalPhisym_cfi.py,v 1.4 2009/09/23 16:05:56 kodolova Exp $
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
    rawInputLabel=cms.InputTag("source"),
    period = cms.uint32(4096),                               
    # File to save 
    SaveToFile = cms.untracked.bool(False),
    FileName = cms.untracked.string('MonitorAlCaHcalPhiSym.root'),
    # DQM folder to write to
    FolderName = cms.untracked.string('AlCaReco/HcalPhiSym')
)



