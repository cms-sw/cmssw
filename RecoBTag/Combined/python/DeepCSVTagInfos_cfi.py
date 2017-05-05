import FWCore.ParameterSet.Config as cms
from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import combinedSecondaryVertexCommon

DeepCSVTagInfos = cms.EDProducer(
	'TrackDeepNNTagInfoProducer',
	svTagInfos = cms.InputTag('inclusiveSecondaryVertexFinderTagInfos'),
	computer = combinedSecondaryVertexCommon
	)
