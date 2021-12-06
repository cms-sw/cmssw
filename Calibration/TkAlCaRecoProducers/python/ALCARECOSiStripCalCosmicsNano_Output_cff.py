import FWCore.ParameterSet.Config as cms

# AlCaNano for track based calibration using Cosmics events
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import NANOAODEventContent
OutALCARECOSiStripCalCosmicsNano_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripCalCosmicsNano')
    ),
    outputCommands=cms.untracked.vstring(
        'keep nanoaodFlatTable_*Table_*_*',
        'keep nanoaodMergeableCounterTable_*Table_*_*',
        'keep nanoaodUniqueString_nanoMetadata_*_*',
        )
    )
import copy
OutALCARECOSiStripCalCosmicsNano = copy.deepcopy(OutALCARECOSiStripCalCosmicsNano_noDrop)
OutALCARECOSiStripCalCosmicsNano.outputCommands.insert(0, "drop *")
