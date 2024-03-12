import FWCore.ParameterSet.Config as cms
from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import combinedSecondaryVertexCommon

pfDeepCSVTagInfos = cms.EDProducer(
	'DeepNNTagInfoProducer',
	svTagInfos = cms.InputTag('pfInclusiveSecondaryVertexFinderTagInfos'),
	computer = combinedSecondaryVertexCommon
	)
# foo bar baz
# eJF8SN0MqMz3M
