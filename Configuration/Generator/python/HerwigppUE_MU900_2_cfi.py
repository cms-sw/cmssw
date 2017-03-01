import FWCore.ParameterSet.Config as cms

herwigppUESettingsBlock = cms.PSet(

	hwpp_ue_MU900_2 = cms.vstring(
		'+hwpp_pdf_MRST2008LOss',
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 900.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.2*GeV',
		# Colour reconnection settings
		'set /Herwig/Hadronization/ColourReconnector:ColourReconnection Yes',
		'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.4114451',
		# Colour Disrupt settings
		'set /Herwig/Partons/RemnantDecayer:colourDisrupt 0.2977505',
		# inverse hadron radius
		'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.154816',
		# MPI model settings
		'set /Herwig/UnderlyingEvent/KtCut:MinKT 2.599215',
		'set /Herwig/UnderlyingEvent/UECuts:MHatMin 5.19843',
		#'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
		'set /Herwig/UnderlyingEvent/MPIHandler:softInt Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:twoComp Yes',
		'set /Herwig/UnderlyingEvent/MPIHandler:DLmode 3',
	),
)
