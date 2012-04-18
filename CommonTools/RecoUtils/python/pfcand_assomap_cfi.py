import FWCore.ParameterSet.Config as cms

PFCandAssoMap = cms.EDProducer('PFCand_AssoMap',		
	 
	  #Set the Input Collections
          PFCandidateCollection = cms.InputTag('particleFlow'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
		  
	  #Set the Input Association Map
          VertexTrackAssociationMap = cms.InputTag('Tracks2Vertex'),
		  	    
	  #Configuration for the reassociation of gamma conversion particles
          ConversionsCollection = cms.InputTag('allConversions'),
	   
	  #Configuration for the reassociation of particles from V0 decays
          V0KshortCollection = cms.InputTag('generalV0Candidates','Kshort'),
          V0LambdaCollection = cms.InputTag('generalV0Candidates','Lambda'),
	   
	  #Configuration for the reassociation of particles from nuclear interactions
          NIVertexCollection = cms.InputTag('particleFlowDisplacedVertex'),
)

