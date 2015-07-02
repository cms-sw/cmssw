import FWCore.ParameterSet.Config as cms

# Reshuffling using MostOffShell option
# All particles put on the new-mass shell and then the most off-shell and recoiling system are rescaled to ensure 4-momentum is conserved.

herwigppReshuffleSettingsBlock = cms.PSet(

	hwpp_reshuffle_MostOffShell = cms.vstring(
		'set /Herwig/Shower/KinematicsReconstructor:FinalStateReconOption MostOffShell',
	),
)

