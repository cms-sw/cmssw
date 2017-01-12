import FWCore.ParameterSet.Config as cms
from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import combinedSecondaryVertexCommon

deepNNTagInfos = cms.EDProducer(
	'DeepNNTagInfoProducer',
	svTagInfos = cms.InputTag('pfInclusiveSecondaryVertexFinderTagInfos'),
	computer = combinedSecondaryVertexCommon
	)
