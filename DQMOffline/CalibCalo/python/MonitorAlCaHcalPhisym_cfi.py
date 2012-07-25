# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
# $Id: MonitorAlCaHcalPhisym_cfi.py,v 1.5 2009/11/16 12:31:38 kodolova Exp $
#
# \author Stefano Argiro
#
HcalPhiSymMon = cms.EDAnalyzer("DQMHcalPhiSymAlCaReco",
    # product to monitor
    hbheInputMB = cms.InputTag("hbheprereco"),
    hoInputMB = cms.InputTag("horeco"),
    hfInputMB = cms.InputTag("hfreco"),
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



