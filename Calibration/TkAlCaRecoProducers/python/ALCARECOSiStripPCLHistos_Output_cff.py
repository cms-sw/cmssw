import FWCore.ParameterSet.Config as cms

# AlCaReco for Bad Component Identification
OutALCARECOSiStripPCLHistos_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripPCLHistos')
    ),
    outputCommands=cms.untracked.vstring(   
        'keep *_MEtoEDMConvertSiStrip_*_*')
)

import copy
OutALCARECOSiStripPCLHistos=copy.deepcopy(OutALCARECOSiStripPCLHistos_noDrop)
OutALCARECOSiStripPCLHistos.outputCommands.insert(0,"drop *")
