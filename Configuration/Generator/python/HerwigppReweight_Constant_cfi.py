import FWCore.ParameterSet.Config as cms

# Reweight using constant

herwigppReweightSettingsBlock = cms.PSet(

	hwpp_reweight_Constant = cms.vstring(
		'mkdir /Herwig/Weights',
		'create ThePEG::ReweightConstant /Herwig/Weights/reweightConstant ReweightConstant.so',
		'set /Herwig/Weights/reweightConstant:C 1',
		'insert /Herwig/MatrixElements/SimpleQCD:Reweights[0] /Herwig/Weights/reweightConstant',
	),
)

