import FWCore.ParameterSet.Config as cms

AssociationMaps = cms.EDProducer('PF_PU_AssoMap',			
	 
	  #Choose which map should be created
	  #"VertexToTracks", "TracksToVertex" or "Both"
	  AssociationType = cms.InputTag('Both'),		
	 
	  #Set the number of associations per track/vertex		   	    
          MaxNumberOfAssociations = cms.int32(1),	
	 
	  #Set the Input Collections
          TrackCollection = cms.InputTag('generalTracks'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
	    
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
		  
	  #Configuration for the final association
	  # 0 == always first vertex (default)
	  # 1 == closest vertex in z/longitudinal distance
	  # 2 == closest vertex in 3D
          FinalAssociation = cms.untracked.int32(1),			    
	   	   
	  #What to do if the dipl vertex coll can't be found
          ignoreMissingCollection = cms.bool(True),			    
	   	   
	  #Input for the search of the closest vertex
          nTrackWeight = cms.double(0.001),
		  
)

AssociationMapsJetMet = cms.EDProducer('PF_PU_AssoMap',		
	 
	  #Choose which map should be created
	  # "VertexToTracks", "TracksToVertex" or "Both"
	  AssociationType = cms.InputTag('Both'),		
	 
	  #Set the number of associations per track/vertex		   	    
          MaxNumberOfAssociations = cms.int32(1),			
	 
	  #Set the Input Collections
          TrackCollection = cms.InputTag('generalTracks'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
	    
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

AssociationMapsMuoEgm = cms.EDProducer('PF_PU_AssoMap',			
	 
	  #Choose which map should be created
	  # "VertexToTracks", "TracksToVertex" or "Both"
	  AssociationType = cms.InputTag('Both'),		
	 
	  #Set the number of associations per track/vertex		   	    
          MaxNumberOfAssociations = cms.int32(1),		
	 
	  #Set the Input Collections
          TrackCollection = cms.InputTag('generalTracks'),
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
	    
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

#Old configuration for imports from other packages
#to avoid undefined references
Tracks2Vertex =	AssociationMaps.clone()		
	 
	  #Choose which map should be created
	  #"VertexToTracks", "TracksToVertex" or "Both"
	  AssociationType = cms.InputTag('TracksToVertex'),
	 
)
