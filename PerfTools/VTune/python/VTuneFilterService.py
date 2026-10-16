import FWCore.ParameterSet.Config as cms

def customise(process):
    # Load your custom profiling controller
    process.VTuneFilterService = cms.Service("VTuneFilterService",
        # Pass an explicit list of module labels you want to ENABLE profiling for
        moduleNames = cms.untracked.vstring(
        '',
        ),
        # or use the allModules option to enable profiling for all modules
        allModules = cms.untracked.bool(True)
        )
    return (process)
