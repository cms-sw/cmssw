import FWCore.ParameterSet.Config as cms

herwigppUESettingsBlock = cms.PSet(

	hwpp_ue_7_2 = cms.vstring(
		'+hwpp_pdf_MRST2008LOss',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 7000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.2*GeV',
		# Colour reconnection settings
		'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
		'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.6165547',
		# Colour Disrupt settings
		'set /Herwig/Partons/RemnantDecayer:colourDisrupt 0.3493643',
		# inverse hadron radius
		'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 0.81',
		# MPI model settings
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 3.36',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 6.72',
		#'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
		'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 3',
	),
)
