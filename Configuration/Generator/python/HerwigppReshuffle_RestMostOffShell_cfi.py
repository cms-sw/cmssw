import FWCore.ParameterSet.Config as cms

# Reshuffling using RestMostOffShell option
# The most off-shell is put on shell by rescaling it and the recoiling system, the recoiling system is then put on-shell in its rest frame.

herwigppReshuffleSettingsBlock = cms.PSet(

	hwpp_reshuffle_RestMostOffShell = cms.vstring(
		'set /Herwig/Shower/KinematicsReconstructor:FinalStateReconOption RestMostOffShell',
	),
)

