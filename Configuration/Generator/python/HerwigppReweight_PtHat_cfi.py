import FWCore.ParameterSet.Config as cms

# Reweight using pTHat

herwigReweightSettingsBlock = cms.PSet(

	reweightPthat = cms.vstring(
		'mkdir /Herwig/Weights',
		'cd /Herwig/Weights',
		'create ThePEG::ReweightMinPT reweightMinPT ReweightMinPT.so',
		'cd /',
		'set /Herwig/Weights/reweightMinPT:Power 4.5',
		'set /Herwig/Weights/reweightMinPT:Scale 15*GeV',
		'insert SimpleQCD:Reweights[0] /Herwig/Weights/reweightMinPT',
	),
)

