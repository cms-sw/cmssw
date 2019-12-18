import FWCore.ParameterSet.Config as cms

herwig7PSWeightsSettingsBlock = cms.PSet(
    hw_PSWeights_settings = cms.vstring(
        'cd /',
        'cd /Herwig/Shower',
        'do ShowerHandler:AddVariation RedHighAll 1.141 1.141  All',
        'do ShowerHandler:AddVariation RedLowAll 0.707 0.707 All',
        'do ShowerHandler:AddVariation DefHighAll 2 2 All',
        'do ShowerHandler:AddVariation DefLowAll 0.5 0.5 All',
        'do ShowerHandler:AddVariation ConHighAll 4 4 All',
        'do ShowerHandler:AddVariation ConLowAll 0.25 0.25 All',
        'do ShowerHandler:AddVariation RedHighHard 1.141 1.141  Hard',
        'do ShowerHandler:AddVariation RedLowHard 0.707 0.707 Hard',
        'do ShowerHandler:AddVariation DefHighHard 2 2 Hard',
        'do ShowerHandler:AddVariation DefLowHard 0.5 0.5 Hard',
        'do ShowerHandler:AddVariation ConHighHard 4 4 Hard',
        'do ShowerHandler:AddVariation ConLowHard 0.25 0.25 Hard',
        'do ShowerHandler:AddVariation RedHighSecondary 1.141 1.141  Secondary',
        'do ShowerHandler:AddVariation RedLowSecondary 0.707 0.707 Secondary',
        'do ShowerHandler:AddVariation DefHighSecondary 2 2 Secondary',
        'do ShowerHandler:AddVariation DefLowSecondary 0.5 0.5 Secondary',
        'do ShowerHandler:AddVariation ConHighSecondary 4 4 Secondary',
        'do ShowerHandler:AddVariation ConLowSecondary 0.25 0.25 Secondary',
        'set SplittingGenerator:Detuning 2.0',
        'cd /',
    )
)
