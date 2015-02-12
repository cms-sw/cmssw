import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

candidateCombinedSecondaryVertexSoftLeptonCtagLComputer = cms.ESProducer("CandidateCombinedSecondaryVertexSoftLeptonCtagLESProducer",
	combinedSecondaryVertexCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'CombinedSVRecoVertexNoSoftLeptonCtagL', 
		'CombinedSVPseudoVertexNoSoftLeptonCtagL', 
		'CombinedSVNoVertexNoSoftLeptonCtagL',
		'CombinedSVRecoVertexSoftMuonCtagL', 
		'CombinedSVPseudoVertexSoftMuonCtagL', 
		'CombinedSVNoVertexSoftMuonCtagL',
		'CombinedSVRecoVertexSoftElectronCtagL', 
		'CombinedSVPseudoVertexSoftElectronCtagL', 
		'CombinedSVNoVertexSoftElectronCtagL'),
	categoryVariableName = cms.string('vertexLeptonCategory')
)

