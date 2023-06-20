import FWCore.ParameterSet.Config as cms

# Settings from $HERWIGPATH/LHE-MCatNLO.in, should be used together with the Herwig7LHECommonSettings

herwig7LHEMG5aMCatNLOSettingsBlock = cms.PSet(
    hw_lhe_MG5aMCatNLO_settings = cms.vstring(
        'set /Herwig/Shower/KinematicsReconstructor:InitialInitialBoostOption LongTransBoost',
        'set /Herwig/Shower/KinematicsReconstructor:ReconstructionOption General',
        'set /Herwig/Shower/KinematicsReconstructor:InitialStateReconOption Rapidity',
        'set /Herwig/Shower/ShowerHandler:SpinCorrelations Yes',
        'set /Herwig/Particles/t:NominalMass 172.5'
    )
)
