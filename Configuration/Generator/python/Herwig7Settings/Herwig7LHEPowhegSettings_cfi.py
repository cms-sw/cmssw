import FWCore.ParameterSet.Config as cms

# Settings from $HERWIGPATH/LHE-POWHEG.in, should be used together with the Herwig7LHECommonSettings

herwig7LHEPowhegSettingsBlock = cms.PSet(
    hw_lhe_powheg_settings = cms.vstring(
        'set /Herwig/Shower/ShowerHandler:MaxPtIsMuF Yes',
        'set /Herwig/Shower/ShowerHandler:RestrictPhasespace Yes',
        'set /Herwig/Shower/PartnerFinder:PartnerMethod Random',
        'set /Herwig/Shower/PartnerFinder:ScaleChoice Partner',
        'set /Herwig/Particles/t:NominalMass 172.5'
    )
)
