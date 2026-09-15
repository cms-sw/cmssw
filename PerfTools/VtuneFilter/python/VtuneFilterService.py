import FWCore.ParameterSet.Config as cms

def customise(process):
    # Load your custom profiling controller
    process.VtuneFilterService = cms.Service("VtuneFilterService",
        # Pass an explicit list of module labels you want to DISABLE profiling for
        targetModules = cms.untracked.vstring(
            'deepTau2018v2p5ForMini',
            'deepTau2026v2p5ForMini',
            'boostedDeepTau20161718v2p0BoostedForMini',
        )
    )
    return (process)
