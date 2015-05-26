import FWCore.ParameterSet.Config as cms

# Tune in Herwig version 2.7 for the CTEQ6L1 PDF
# UE-EE-5-CTEQ6L1, see
# https://herwig.hepforge.org/trac/wiki/MB_UE_tunes

herwigppUESettingsBlock = cms.PSet(

     hwpp_ue_EE5CEnergyExtrapol =  cms.vstring(
        'set /Herwig/UnderlyingEvent/MPIHandler:EnergyExtrapolation Power',
        'set /Herwig/UnderlyingEvent/MPIHandler:ReferenceScale 7000.*GeV',
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.33',
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.91*GeV',
        ),

     hwpp_ue_EE5C =  cms.vstring(
        '+hwpp_ue_EE5CEnergyExtrapol',
        # Colour reconnection settings
        'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
        'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.49',
        # Colour Disrupt settings
        'set /Herwig/Partons/RemnantDecayer:colourDisrupt 0.80',
        # inverse hadron radius
        'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 2.30',
        # MPI model settings
        'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
        'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
        'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 2',
        ),
)
