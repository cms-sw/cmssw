import FWCore.ParameterSet.Config as cms

herwigppUESettingsBlock = cms.PSet(
	herwigppUE_EE_2_Base = cms.vstring(
		'+pdfMRST2008LOss',
		# Energy-independent MPI parameters
		#   Colour reconnection settings
		'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
		'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.55',
		#   Colour Disrupt settings
		'set /Herwig/Partons/RemnantDecayer:colourDisrupt 0.15',
		#   inverse hadron radius
		'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.1',
		#   MPI model settings
		#'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
		'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 3',
		'cd /',
	),

	herwigppUE_EE_2_900GeV = cms.vstring(
		'+herwigppUE_EE_2_Base',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 900.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 2.34',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 4.68',
	),

	herwigppUE_EE_2_1800GeV = cms.vstring(
		'+herwigppUE_EE_2_Base',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 1800.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 3.09',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 6.18',
	),

	herwigppUE_EE_2_2760GeV = cms.vstring(
		'+herwigppUE_EE_2_Base',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 2760.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 3.31',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 6.62',
	),

	herwigppUE_EE_2_7000GeV = cms.vstring(
		'+herwigppUE_EE_2_Base',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 7000.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 4.02',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 8.04',
	),
)
