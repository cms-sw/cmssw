# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: MonitorAlCaHcalPhisym_cfi.py,v 1.7 2012/10/16 17:18:50 safronov Exp $
#
# \author Stefano Argiro
#
HcalPhiSymMon = cms.EDAnalyzer("DQMHcalPhiSymAlCaReco",
    # product to monitor
    hbheInputMB = cms.InputTag("hbherecoMB"),
    hoInputMB = cms.InputTag("horecoMB"),
    hfInputMB = cms.InputTag("hfrecoMBspecial"),
    hbheInputNoise = cms.InputTag("hbherecoNoise"),
    hoInputNoise = cms.InputTag("horecoNoise"),
    hfInputNoise = cms.InputTag("hfrecoNoise"),
    rawInputLabel=cms.InputTag("rawDataCollector"),
    period = cms.uint32(4096),                               
    # File to save 
    SaveToFile = cms.untracked.bool(False),
    FileName = cms.untracked.string('MonitorAlCaHcalPhiSym.root'),
    # DQM folder to write to
    FolderName = cms.untracked.string('AlCaReco/HcalPhiSym')
)



