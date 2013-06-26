import FWCore.ParameterSet.Config as cms

FirstVertexPFCandidates = cms.EDProducer('PFCand_NoPU_WithAM',				
	 
	  #Choose which map should be created
	  # "VertexToPFCands", "PFCandsToVertex" or "Both"
	  AssociationType = cms.InputTag('Both'),	
		  
	  #Set the Input Association Map
          VertexPFCandAssociationMap = cms.InputTag('PFCandAssoMap'),
	 
	  #Set the Input Collections
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
	 
	  #Set the minimum quality of the association
          MinQuality = cms.int32(2),
)

