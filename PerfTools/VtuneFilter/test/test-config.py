import FWCore.ParameterSet.Config as cms

process = cms.Process("VTUNEFILTER")

# Load your custom profiling controller
process.VtuneFilterService = cms.Service("VtuneFilterService",
    # Pass an explicit list of module labels you want to ENABLE profiling for
    targetModules = cms.untracked.vstring(
        'demoAnalyzerModule', 
        'specificMuonProducer'
    )
)
