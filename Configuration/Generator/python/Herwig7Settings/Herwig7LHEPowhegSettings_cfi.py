import FWCore.ParameterSet.Config as cms

herwig7LHEPowhegSettingsBlock = cms.PSet(
    hw_lhe_Powheg_settings = cms.vstring(
        'set /Herwig/Shower/ShowerHandler:MaxPtIsMuF Yes',
        'set /Herwig/Shower/ShowerHandler:RestrictPhasespace Yes',
        'set /Herwig/Shower/PartnerFinder:PartnerMethod Random',
        'set /Herwig/Shower/PartnerFinder:ScaleChoice Partner',
        'set /Herwig/Shower/GtoQQbarSplitFn:AngularOrdered Yes',
        'set /Herwig/Shower/GammatoQQbarSplitFn:AngularOrdered Yes',
        'set /Herwig/Particles/t:NominalMass 172.5'
    )
)
