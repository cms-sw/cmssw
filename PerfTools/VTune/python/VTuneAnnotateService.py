import FWCore.ParameterSet.Config as cms

def customise(process):
    # Load your custom profiling controller
    process.VTuneAnnotateService = cms.Service("VTuneAnnotateService",
        # Pass an explicit list of module labels you want to ENABLE annotation for
        moduleNames = cms.untracked.vstring(
            '',
        ),
        # or use the allModules option to enable annotation for all modules
        allModules = cms.untracked.bool(True),
        # The ITT domain to use for annotation
        ittDomain = cms.untracked.string("CMSSW.ModuleTracker")
        )
    return(process)
