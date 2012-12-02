import FWCore.ParameterSet.Config as cms

FirstVertexPFCandidates = cms.EDProducer('PFCand_NoPU_WithAM',		
		  
	  #Set the Input Association Map
          VertexPFCandAssociationMap = cms.InputTag('PFCandAssoMap'),
)

