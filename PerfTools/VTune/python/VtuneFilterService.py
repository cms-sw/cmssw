import FWCore.ParameterSet.Config as cms

def customise(process):
    # Load your custom profiling controller
    process.VtuneFilterService = cms.Service("VtuneFilterService",
        # Pass an explicit list of module labels you want to ENABLE profiling for
        targetModules = cms.untracked.vstring(
            'all', # Use 'all' to enable profiling for all modules
        )
    )
    return (process)
