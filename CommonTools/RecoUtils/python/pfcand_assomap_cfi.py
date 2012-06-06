import FWCore.ParameterSet.Config as cms

PFCandAssoMap = cms.EDProducer('PFCand_AssoMap',		
	 
	  #Set the Input Collections
          PFCandidateCollection = cms.InputTag('particleFlow'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
		  	    
	  #Configuration for the reassociation of gamma conversion particles
          ConversionsCollection = cms.InputTag('allConversions'),
	   
	  #Configuration for the reassociation of particles from V0 decays
          V0KshortCollection = cms.InputTag('generalV0Candidates','Kshort'),
          V0LambdaCollection = cms.InputTag('generalV0Candidates','Lambda'),
	   
	  #Configuration for the reassociation of particles from nuclear interactions
          NIVertexCollection = cms.InputTag('particleFlowDisplacedVertex'),
	   	   
	  #Configuration to check if a secondary is compatible with the BeamSpot
	  #True for best performance in jet-pt response
	  UseBeamSpotCompatibility = cms.untracked.bool(True),
	  BeamSpot = cms.InputTag('offlineBeamSpot'),
		  
	  #Configuration for the final association
          VertexAssClosest = cms.untracked.bool(True),			    
	   	   
	  #What to do if the dipl vertex coll can't be found
          ignoreMissingCollection = cms.bool(True),	
		  
)
