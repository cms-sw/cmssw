import FWCore.ParameterSet.Config as cms

def customise(process):
    # Load your custom profiling controller
    process.VtuneAnnontateService = cms.Service("VtuneAnnotateService",
        # Pass an explicit list of module labels you want to ENABLE annotation for
        targetModules = cms.untracked.vstring(
            'deepTau2018v2p5ForMini',
        )
    )
    return(process)
