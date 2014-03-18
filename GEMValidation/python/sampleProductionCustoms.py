import FWCore.ParameterSet.Config as cms

def dropNonMuonCollections(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep  *_*_*_*')
            # drop tracker simhits
            getattr(process,b).outputCommands.append('drop PSimHits_*_Tracker*_*')
            # drop calorimetry stuff
            getattr(process,b).outputCommands.append('drop PCaloHits_*_*_*')
            # clean up simhits from other detectors
            getattr(process,b).outputCommands.append('drop PSimHits_*_Totem*_*')
            getattr(process,b).outputCommands.append('drop PSimHits_*_FP420*_*')
            getattr(process,b).outputCommands.append('drop PSimHits_*_BSC*_*')
            getattr(process,b).outputCommands.append('drop *RandomEngineStates_*_*_*')
            getattr(process,b).outputCommands.append('drop *_randomEngineStateProducer_*_*')
    return process

def keepMuonTrackVertexCollections(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('drop *_*_*_*')
            getattr(process,b).outputCommands.append('keep SimTracks_*_*_*')
            getattr(process,b).outputCommands.append('keep SimVertexs_*_*_*')
            getattr(process,b).outputCommands.append('keep *_*_MuonCSCHits_*')
            getattr(process,b).outputCommands.append('keep *_*_MuonRPCHits_*')
            getattr(process,b).outputCommands.append('keep *_*_MuonDTHits_*')
            getattr(process,b).outputCommands.append('keep *_*_MuonGEMHits_*')
            getattr(process,b).outputCommands.append('keep *_*_MuonME0Hits_*')
    return process
