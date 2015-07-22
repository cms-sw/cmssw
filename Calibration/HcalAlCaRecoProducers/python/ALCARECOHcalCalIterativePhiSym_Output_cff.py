import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Iterative Phi Syms
# output module 
#  module alcastreamHcalCalIterativePhiSymOutput = PoolOutputModule

OutALCARECOHcalCalIterativePhiSym_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIterativePhiSym')
    ),
    outputCommands = cms.untracked.vstring(

                                            "keep *_horeco_*_*",
                                            "keep *_hfreco_*_*",
                                            "keep *_hbhereco_*_*",
                                            "keep *_offlinePrimaryVertices_*_*",
                                            "keep edmTriggerResults_*_*_HLT")

)

import copy
OutALCARECOHcalCalIterativePhiSym=copy.deepcopy(OutALCARECOHcalCalIterativePhiSym_noDrop)
OutALCARECOHcalCalIterativePhiSym.outputCommands.insert(0, "drop *")
