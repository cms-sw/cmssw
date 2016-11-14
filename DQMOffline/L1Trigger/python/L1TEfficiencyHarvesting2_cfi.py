import FWCore.ParameterSet.Config as cms

l1tEfficiencyHarvesting = cms.EDAnalyzer(
    "L1TEfficiencyHarvesting",
    verbose=cms.untracked.bool(False),
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(
            numeratorDir=cms.untracked.string("L1T/Efficiency/Muons"),
            plots=cms.untracked.vstring(
                "EffvsPt16", "EffvsEta16", "EffvsPhi16",
                "EffvsPt20", "EffvsEta20", "EffvsPhi20",
                "EffvsPt25", "EffvsEta25", "EffvsPhi25",
            )
        )
    )

)
