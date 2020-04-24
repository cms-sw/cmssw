import FWCore.ParameterSet.Config as cms

# Default tune in Herwig version 2.7 UE-EE-5 (MRST LO??), see
# https://herwig.hepforge.org/trac/wiki/MB_UE_tunes

herwigppUESettingsBlock = cms.PSet(

     hwpp_ue_EE5EnergyExtrapol =  cms.vstring(
        'set /Herwig/UnderlyingEvent/MPIHandler:EnergyExtrapolation Power',
        'set /Herwig/UnderlyingEvent/MPIHandler:ReferenceScale 7000.*GeV',
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.314',
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 4.620*GeV',
        ),

     hwpp_ue_EE5 =  cms.vstring(
        '+hwpp_ue_EE5EnergyExtrapol',
        # Colour reconnection settings
        'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
        'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.420',
        # Colour Disrupt settings
        'set /Herwig/Partons/RemnantDecayer:colourDisrupt 0.860',
        # inverse hadron radius
        'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 2.240',
        # MPI model settings
        'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
        'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
        'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 2',
        ),
)
