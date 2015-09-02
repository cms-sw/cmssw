import FWCore.ParameterSet.Config as cms

# Reshuffling using Recursive option
# Recursively put on shell by putting the most off-shell particle which hasn't been rescaled on-shell by rescaling the particles and the recoiling system.

herwigppReshuffleSettingsBlock = cms.PSet(

	hwpp_reshuffle_Recursive = cms.vstring(
		'set /Herwig/Shower/KinematicsReconstructor:FinalStateReconOption Recursive',
	),
)

