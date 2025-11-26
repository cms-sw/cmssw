import FWCore.ParameterSet.Config as cms

def customise_rawSecond(process):

    # Loop over real output modules
    for name, module in process.outputModules.items():
        if hasattr(module, "outputCommands"):
            cmds = list(module.outputCommands)
            new_cmds = [
                cmd.replace("hltSiStripClusters2ApproxClusters",
                            "hltSiStripClusters2ApproxClustersv1")
                for cmd in cmds
            ]
            module.outputCommands = cms.untracked.vstring(*new_cmds)
    
    if hasattr(process, "siStripClusters"):
        process.siStripClusters.v1 = cms.bool(True)

    return process

