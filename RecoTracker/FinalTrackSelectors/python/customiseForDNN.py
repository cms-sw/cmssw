import FWCore.ParameterSet.Config as cms



def customiseForDNN(process):

	process.tf_dummy_source = cms.ESSource("EmptyESSource", recordName = cms.string("TfGraphRecord"), firstValid = cms.vuint32(1), iovIsRunNotTime = cms.bool(True) )
	return process
