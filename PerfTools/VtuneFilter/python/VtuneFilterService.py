import FWCore.ParameterSet.Config as cms

def customise(process):
    # Load your custom profiling controller
    process.VtuneFilterService = cms.Service("VtuneFilterService",
        # Pass an explicit list of module labels you want to DISABLE profiling for
        targetModules = cms.untracked.vstring(
            'demoAnalyzerModule', 
            'specificMuonProducer'
        )
    )
    return (process)
