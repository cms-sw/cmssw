import FWCore.ParameterSet.Config as cms

OutALCARECOPPSCalTrackBasedSel_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPPSCalTrackBasedSel')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOPPSCalTrackBasedSel_*_*',
        'keep *_ctppsDiamondRawToDigi_*_*'
        # will be updated to add the spatial alignment required collections
    )
)

import copy
OutALCARECOPPSCalTrackBasedSel = copy.deepcopy(OutALCARECOPPSCalTrackBasedSel_noDrop)
OutALCARECOPPSCalTrackBasedSel.outputCommands.insert(0, 'drop *')
