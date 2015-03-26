import FWCore.ParameterSet.Config as cms

herwigppUESettingsBlock = cms.PSet(

     EE5CEnergyExtrapol =  cms.vstring(
        'set /Herwig/UnderlyingEvent/MPIHandler:EnergyExtrapolation Power',
        'set /Herwig/UnderlyingEvent/MPIHandler:ReferenceScale 7000.*GeV',
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.33',
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.91*GeV',
        ),

      EE5C =  cms.vstring(
        '+pdfCTEQ6L1',
        '+EE5CEnergyExtrapol',
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
