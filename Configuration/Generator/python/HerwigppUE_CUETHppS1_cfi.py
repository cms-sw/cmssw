import FWCore.ParameterSet.Config as cms

# Tune from CMS for the CTEQ6L1 PDF
# CUETHppS1
# Values provided by Paolo Gunnellini

herwigppUESettingsBlock = cms.PSet(

     hwpp_ue_CUETHppS1EnergyExtrapol =  cms.vstring(
        'set /Herwig/UnderlyingEvent/MPIHandler:EnergyExtrapolation Power',
        'set /Herwig/UnderlyingEvent/MPIHandler:ReferenceScale 7000.*GeV',
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.371',
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.91*GeV',
        ),

     hwpp_ue_CUETHppS1 =  cms.vstring(
        '+hwpp_ue_CUETHppS1EnergyExtrapol',
        # Colour reconnection settings
        'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
        'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.528',
        # Colour Disrupt settings
        'set /Herwig/Partons/RemnantDecayer:colourDisrupt 0.628',
        # inverse hadron radius
        'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 2.255',
        # MPI model settings
        'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
        'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
        'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 2',
        ),
)
