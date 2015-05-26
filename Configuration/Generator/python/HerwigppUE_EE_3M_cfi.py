import FWCore.ParameterSet.Config as cms

herwigppUESettingsBlock = cms.PSet(

	hwpp_ue_EE_3M_Common = cms.vstring(
		'+hwpp_pdf_MRST2008LOss',

		'create Herwig::O2AlphaS /Herwig/O2AlphaS',
		'set Model:QCD/RunningAlphaS /Herwig/O2AlphaS',

		# Energy-independent MPI parameters
		#   Colour reconnection settings
		'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
		'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.61',
		#   Colour Disrupt settings
		'set /Herwig/Partons/RemnantDecayer:colourDisrupt 0.75',
		#   inverse hadron radius
		'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.35',
		#   MPI model settings
		'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 2',
	),

	hwpp_ue_EE_3M_900GeV = cms.vstring(
		'+herwigpp_ue_EE_3M_Common',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 900.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 1.86',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 3.72',
	),

	hwpp_ue_EE_3M_1800GeV = cms.vstring(
		'+herwigpp_ue_EE_3M_Common',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 1800.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 2.55',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 5.1',
	),

	hwpp_ue_EE_3M_2760GeV = cms.vstring(
		'+herwigpp_ue_EE_3M_Common',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 2760.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 2.62',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 5.24',
	),

	hwpp_ue_EE_3M_7000GeV = cms.vstring(
		'+herwigpp_ue_EE_3M_Common',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 7000.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 3.06',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 6.12',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.37*GeV',
	),

	hwpp_ue_EE_3M_8000GeV = cms.vstring(
		'+hwpp_ue_EE_3M_Common',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 8000.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 3.21',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 6.42',
	),

	hwpp_ue_EE_3M_14000GeV = cms.vstring(
		'+hwpp_ue_EE_3M_Common',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 14000.0',
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 3.53',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 7.06',
	),
)
