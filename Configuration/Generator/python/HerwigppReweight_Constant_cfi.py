import FWCore.ParameterSet.Config as cms

# Reweight using constant

herwigppReweightSettingsBlock = cms.PSet(

	hwpp_reweight_Constant = cms.vstring(
		'mkdir /Herwig/Weights',
		'cd /Herwig/Weights',
		'create ThePEG::ReweightConstant reweightConstant ReweightConstant.so',
		'cd /',
		'set /Herwig/Weights/reweightConstant:C 1',
		'insert SimpleQCD:Reweights[0] /Herwig/Weights/reweightConstant',
	),
)

