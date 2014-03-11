import FWCore.ParameterSet.Config as cms

def dropNonMuonCollections(process):
    process.FEVTDEBUGoutput.outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
        # drop tracker simhits
        'drop PSimHits_*_Tracker*_*',
        # drop calorimetry stuff
        'drop PCaloHits_*_*_*',
        # clean up simhits from other detectors
        'drop PSimHits_*_Totem*_*',
        'drop PSimHits_*_FP420*_*',
        'drop PSimHits_*_BSC*_*',
        'drop *RandomEngineStates_*_*_*',
        'drop *_randomEngineStateProducer_*_*'
    )
    return process

