import FWCore.ParameterSet.Config as cms

# Matrix element correction options
# By default matrix element corrections are switched off
# Note: In Herwig++ 2.7.1 matrix element corrections are switched on by default, however this is not recommend if LHE files are showered
# Requiring the hwpp_MECorr_Common block results that the user must make a choice
# Since the single blocks will be alphabetically ordered, we can override the common block and still require that it exists.

herwigppMECorrectionsSettingsBlock = cms.PSet(

	# Deactive ME corrections at all
	# New standard behaviour
	hwpp_MECorr_Common = cms.vstring(
		'set /Herwig/Shower/Evolver:MECorrMode No',					# Deactive ME corrections at all
	),

	# Wrapper to allow more homogenous syntax
	hwpp_MECorr_Off = cms.vstring(
		'+hwpp_MECorr_Common',								# Deactive ME corrections at all
	),

	# Activate ME corrections for the hard process
	hwpp_MECorr_HardOn = cms.vstring(
		'+hwpp_MECorr_Common',
		'set /Herwig/Shower/Evolver:MECorrMode Hard',					# Activate ME corrections for the hard process
	),

	# Activate ME corrections for the soft process
	hwpp_MECorr_SoftOn = cms.vstring(
		'+hwpp_MECorr_Common',
		'set /Herwig/Shower/Evolver:MECorrMode Soft',					# Activate ME corrections for the soft process
	),

	# Activate ME corrections for the soft and hard process
	hwpp_MECorr_On = cms.vstring(
		'+hwpp_MECorr_Common',
		'set /Herwig/Shower/Evolver:MECorrMode Yes',					# Activate ME corrections for the soft and hard process
	),
)

