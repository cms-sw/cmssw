# last update: $Date: 2010/03/16 17:08:51 $ by $Author: mussgill $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmicsHLT_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTFHLT', 
            'pathALCARECOTkAlCosmicsCosmicTFHLT', 
            'pathALCARECOTkAlCosmicsRegionalHLT')
    ),
    outputCommands = cms.untracked.vstring()
)

# We have the same producers as in the non-HLT path, just HLT sel. in front,
# ==> identical keep statements:
import Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_Output_cff
OutALCARECOTkAlCosmicsHLT_noDrop.outputCommands = Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_Output_cff.OutALCARECOTkAlCosmics_noDrop.outputCommands

import copy
OutALCARECOTkAlCosmicsHLT = copy.deepcopy(OutALCARECOTkAlCosmicsHLT_noDrop)
OutALCARECOTkAlCosmicsHLT.outputCommands.insert(0, "drop *")

