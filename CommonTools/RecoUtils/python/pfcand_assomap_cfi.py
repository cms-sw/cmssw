import FWCore.ParameterSet.Config as cms

PFCandAssoMap = cms.EDProducer('PFCand_AssoMap',		
	 
	  #Set the Input Collections
          PFCandidateCollection = cms.InputTag('particleFlow'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
		 
	  #Step 0: If track-pt() is smaler the track will 
	  #always be associated to the first vertex
          TrackPtCut = cms.double(20.),
	    
	  #Set the BeamSpot
	  BeamSpot = cms.InputTag('offlineBeamSpot'),
		  
	  #Check for tracks from secondary vertices
          doReassociation = cms.bool(True),
          GetCleanedCollections = cms.bool(False),
		  	    
	  #Configuration for the reassociation of gamma conversion particles
          ConversionsCollection = cms.InputTag('allConversions'),
	   
	  #Configuration for the reassociation of particles from V0 decays
          V0KshortCollection = cms.InputTag('generalV0Candidates','Kshort'),
          V0LambdaCollection = cms.InputTag('generalV0Candidates','Lambda'),
	   
	  #Configuration for the reassociation of particles from nuclear interactions
          NIVertexCollection = cms.InputTag('particleFlowDisplacedVertex'),
	   	   
	  #Configuration to check if a secondary is compatible with the BeamSpot
	  UseBeamSpotCompatibility = cms.untracked.bool(False), 
          BeamSpotCompatibilityCut = cms.double(5.), 
		  
	  #Configuration for the final association
	  # 0 == always first vertex (default)
	  # 1 == closest vertex in z/longitudinal distance
	  # 2 == closest vertex in 3D
          FinalAssociation = cms.untracked.int32(2),			    
	   	   
	  #What to do if the dipl vertex coll can't be found
          ignoreMissingCollection = cms.bool(True),			    
	   	   
	  #Input for the search of the closest vertex
          nTrackWeight = cms.double(0.01),
		  
)

PFCandAssoMapJme = cms.EDProducer('PFCand_AssoMap',		
	 
	  #Set the Input Collections
          PFCandidateCollection = cms.InputTag('particleFlow'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
		 
	  #Step 0: If track-pt() is smaler the track will 
	  #always be associated to the first vertex
          TrackPtCut = cms.double(1000.),
	    
	  #Set the BeamSpot
	  BeamSpot = cms.InputTag('offlineBeamSpot'),
		  
	  #Check for tracks from secondary vertices
          doReassociation = cms.bool(False),
          GetCleanedCollections = cms.bool(False),
		  	    
	  #Configuration for the reassociation of gamma conversion particles
          ConversionsCollection = cms.InputTag('allConversions'),
	   
	  #Configuration for the reassociation of particles from V0 decays
          V0KshortCollection = cms.InputTag('generalV0Candidates','Kshort'),
          V0LambdaCollection = cms.InputTag('generalV0Candidates','Lambda'),
	   
	  #Configuration for the reassociation of particles from nuclear interactions
          NIVertexCollection = cms.InputTag('particleFlowDisplacedVertex'),
	   	   
	  #Configuration to check if a secondary is compatible with the BeamSpot
	  UseBeamSpotCompatibility = cms.untracked.bool(False), 
          BeamSpotCompatibilityCut = cms.double(5.), 
		  
	  #Configuration for the final association
	  # 0 == always first vertex (default)
	  # 1 == closest vertex in z/longitudinal distance
	  # 2 == closest vertex in 3D
          FinalAssociation = cms.untracked.int32(0),			    
	   	   
	  #What to do if the dipl vertex coll can't be found
          ignoreMissingCollection = cms.bool(True),			    
	   	   
	  #Input for the search of the closest vertex
          nTrackWeight = cms.double(0.),
		  
)

PFCandAssoMapMuoEgm = cms.EDProducer('PFCand_AssoMap',			
	 
	  #Set the Input Collections
          PFCandidateCollection = cms.InputTag('particleFlow'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
		 
	  #Step 0: If track-pt() is smaler the track will 
	  #always be associated to the first vertex
          TrackPtCut = cms.double(1000.),
	    
	  #Set the BeamSpot
	  BeamSpot = cms.InputTag('offlineBeamSpot'),
		  
	  #Check for tracks from secondary vertices
          doReassociation = cms.bool(False),
          GetCleanedCollections = cms.bool(False),
		  	    
	  #Configuration for the reassociation of gamma conversion particles
          ConversionsCollection = cms.InputTag('allConversions'),
	   
	  #Configuration for the reassociation of particles from V0 decays
          V0KshortCollection = cms.InputTag('generalV0Candidates','Kshort'),
          V0LambdaCollection = cms.InputTag('generalV0Candidates','Lambda'),
	   
	  #Configuration for the reassociation of particles from nuclear interactions
          NIVertexCollection = cms.InputTag('particleFlowDisplacedVertex'),
	   	   
	  #Configuration to check if a secondary is compatible with the BeamSpot
	  UseBeamSpotCompatibility = cms.untracked.bool(False), 
          BeamSpotCompatibilityCut = cms.double(5.), 
		  
	  #Configuration for the final association
	  # 0 == always first vertex (default)
	  # 1 == closest vertex in z/longitudinal distance
	  # 2 == closest vertex in 3D
          FinalAssociation = cms.untracked.int32(1),			    
	   	   
	  #What to do if the dipl vertex coll can't be found
          ignoreMissingCollection = cms.bool(True),			    
	   	   
	  #Input for the search of the closest vertex
          nTrackWeight = cms.double(0.),
		  
)
