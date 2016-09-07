import FWCore.ParameterSet.Config as cms

# Reshuffling using RestRecursive option
# As 3 but recursive treated the currently most-off shell, only makes a difference if more than 3 partons.

herwigppReshuffleSettingsBlock = cms.PSet(

	hwpp_reshuffle_RestMostOffShell = cms.vstring(
		'set /Herwig/Shower/KinematicsReconstructor:FinalStateReconOption RestRecursive',
	),
)

