import FWCore.ParameterSet.Config as cms

# Reshuffling using default option
# It is not necessary to use this block since it is used by default
# All the momenta are rescaled in the rest frame


herwigppReshuffleSettingsBlock = cms.PSet(

	hwpp_reshuffle_Default = cms.vstring(
		'set /Herwig/Shower/KinematicsReconstructor:FinalStateReconOption Default',
	),
)

