import FWCore.ParameterSet.Config as cms

# CMS CUETHS1 tune based on
# UE-EE-5-CTEQ6L1, see
# https://herwig.hepforge.org/trac/wiki/MB_UE_tunes

herwigppUESettingsBlock = cms.PSet(

     hwpp_ue_CUETHS1EnergyExtrapol =  cms.vstring(
        'set /Herwig/UnderlyingEvent/MPIHandler:EnergyExtrapolation Power',
	'set /Herwig/UnderlyingEvent/MPIHandler:ReferenceScale 7000.GeV',
	'set /Herwig/UnderlyingEvent/MPIHandler:Power 3.705288e-01',
	'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.91GeV',
        ),

     hwpp_ue_CUETHS1 =  cms.vstring(
        '+hwpp_ue_CUETHS1EnergyExtrapol',
        # Colour reconnection settings
	'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
	'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 5.278926e-01',
	# Colour Disrupt settings
	'set /Herwig/Partons/RemnantDecayer:colourDisrupt 6.284222e-01',
	# inverse hadron radius
	'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 2.254998e+00',
	# MPI model settings
	'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
	'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
	'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 2',
	),
)
