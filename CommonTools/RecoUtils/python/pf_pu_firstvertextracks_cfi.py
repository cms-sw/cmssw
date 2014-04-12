import FWCore.ParameterSet.Config as cms

FirstVertexTracks = cms.EDProducer('PF_PU_FirstVertexTracks',			
	 
	  #Choose which map should be created
	  #"VertexToTracks", "TracksToVertex" or "Both"
	  AssociationType = cms.InputTag('Both'),	
	 
	  #The input parameter for the association maps
	  AssociationMap = cms.InputTag('AssociationMaps'),	   	    
	 
	  #The input parameter for the track & vertex collection
	  TrackCollection = cms.InputTag('generalTracks'),	   
	  VertexCollection = cms.InputTag('offlinePrimaryVertices<'),
	   
	  #The minimum quality an association should have 
	  #so that the track is inserted into the track collection
	  MinQuality = cms.int32(2),
)