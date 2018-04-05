import FWCore.ParameterSet.Config as cms

# Reweight using pTHat

herwigppReweightSettingsBlock = cms.PSet(

	hwpp_reweight_Pthat = cms.vstring(
		'mkdir /Herwig/Weights',
		'create ThePEG::ReweightMinPT /Herwig/Weights/reweightMinPT ReweightMinPT.so',
		'set /Herwig/Weights/reweightMinPT:Power 4.5',
		'set /Herwig/Weights/reweightMinPT:Scale 15*GeV',
		'insert /Herwig/MatrixElements/SimpleQCD:Reweights[0] /Herwig/Weights/reweightMinPT',
	),
)

