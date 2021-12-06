# The following comments couldn't be translated into the new config version:

# prescale

import FWCore.ParameterSet.Config as cms

#
#
# \author Stefano Argiro
#
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
HcalPhiSymMon = DQMEDAnalyzer('DQMHcalPhiSymAlCaReco',
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
    #driven by DQMServices/Core/python/DQMStore_cfi.py
    perLSsaving = cms.untracked.bool(False), 
    # DQM folder to write to
    FolderName = cms.untracked.string('AlCaReco/HcalPhiSym')
)



