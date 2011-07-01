# last update: $Date: 2010/03/16 17:08:51 $ by $Author: mussgill $

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Cosmic muon events
OutALCARECOTkAlCosmics0THLT_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0THLT', 
            'pathALCARECOTkAlCosmicsCosmicTF0THLT', 
            'pathALCARECOTkAlCosmicsRegional0THLT')
    ),
    outputCommands = cms.untracked.vstring()
)
# We have the same producers as in the non-HLT path, just HLT sel. in front,
# ==> identical keep statements:
import Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_Output_cff
OutALCARECOTkAlCosmics0THLT_noDrop.outputCommands = Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_Output_cff.OutALCARECOTkAlCosmics0T_noDrop.outputCommands

import copy
OutALCARECOTkAlCosmics0THLT = copy.deepcopy(OutALCARECOTkAlCosmics0THLT_noDrop)
OutALCARECOTkAlCosmics0THLT.outputCommands.insert(0, "drop *")
