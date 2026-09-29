import FWCore.ParameterSet.Config as cms

def customise(process):
    # Load your custom profiling controller
    process.VtuneAnnotateService = cms.Service("VtuneAnnotateService",
        # Pass an explicit list of module labels you want to ENABLE annotation for
        targetModules = cms.untracked.vstring(
            'all'  # Use 'all' to enable annotation for all modules
        )
    )
    return(process)
