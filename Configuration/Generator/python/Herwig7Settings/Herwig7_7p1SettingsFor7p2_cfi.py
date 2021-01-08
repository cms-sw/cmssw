import FWCore.ParameterSet.Config as cms

herwig7p1SettingsFor7p2Block = cms.PSet(
    hw_7p1SettingsFor7p2 = cms.vstring(
        # Recoil scheme. Dot product + veto scheme is new default in 7.2
        # Q2 was default in 7.1
        'read snippets/Tune-Q2.in',

        # Baryonic CR is default in 7.2
        # Plain CR was used in 7.1 and for CH tunes
        'set /Herwig/Hadronization/ColourReconnector:Algorithm Plain',

        # Changes energy extrapolation of pt_min
        # Default in 7.2 is PowerModified, offset/minimum for low com (200 GeV)
        # Power was default in 7.1.
        'set /Herwig/UnderlyingEvent/MPIHandler:EnergyExtrapolation Power',

        # In 7p2, don't allow MB events to start from sea quarks
        # 7p1 allowed this to happen.  Has effect/bias on MB distributions (makes them softer)
        'set /Herwig/MatrixElements/MEMinBias:OnlyValence 0',

        # Soft ladder transverse momentum sampled from gaussian tail below ptmin for first particle
        # Remainder are sampled flat below the pt of the first particle
        # In 7.1, all particles were sampled from gaussian tail, option 0
        'set /Herwig/Partons/RemnantDecayer:PtDistribution 0',

        # ladderMult is now constant wrt energy.
        # Parametrisation cannot be changed back to energy dependent form in Herwig7.1
        #  Set to value for 13 TeV from CH3
        'set /Herwig/Partons/RemnantDecayer:ladderMult 0.63',
        'set /Herwig/Partons/RemnantDecayer:ladderbFactor 0.0',

        # Diffraction ratio from CH3 in 7p1
        # Taken from log files stating diffractive and non-diffractive cross section
        'set /Herwig/UnderlyingEvent/MPIHandler:DiffractiveRatio 0.21',
        )
)