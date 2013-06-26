
full_title = "RecoVertex collections (in RECO and AOD)"

full = {
    '0':['offlinePrimaryVertices', 'reco::VertexCollection ', 'Primary vertex reconstructed using the tracks taken from the generalTracks collection'] ,
    '1':['offlinePrimaryVerticesWithBS', 'reco::VertexCollection ', 'Primary vertex reconstructed using the tracks taken from the generalTracks collection, and imposing the offline beam spot as a constraint in the fit of the vertex position.'] ,
    '2':['offlinePrimaryVerticesFromCosmicTracks', '*', 'No documentation'] ,
    '3':['nuclearInteractionMaker', '*', 'No documentation'],

  # Correction needed, because not matched with Event Content
    '4':['generalV0Candidates:Kshort','reco::VertexCompositeCandidateCollection ','K0S candidate collection reconstructed using the tracks taken from the generalTracks collection'],
    '5':['generalV0Candidates:Lambda','reco::VertexCompositeCandidateCollection ','Lambda0 candidate collection reconstructed using the tracks taken from the generalTracks collection'],
    '6':['offlinePrimaryVerticesFromCTFTracks','reco::VertexCollection','Identical to the offlinePrimaryVertices collection.'],
    '7':['offlinePrimaryVerticesFromCTFTracks','reco::VertexCollection','Primary vertex reconstructed using the CKF tracks, taken from the ctfWithMaterialTracks collection'],
    '8':['offlinePrimaryVerticesFromRSTracks','reco::VertexCollection','Primary vertex reconstructed using the CKF tracks, taken from the rsWithMaterialTracks collection']    
     
}

reco_title = "RecoVertex collections (in RECO only)"

reco = {
    '0':['offlinePrimaryVertices', 'reco::VertexCollection', 'Primary vertex reconstructed using the tracks taken from the generalTracks collection'] ,
    '1':['offlinePrimaryVerticesWithBS', 'reco::VertexCollection ', 'Primary vertex reconstructed using the tracks taken from the generalTracks collection, and imposing the offline beam spot as a constraint in the fit of the vertex position.'] ,
    '2':['offlinePrimaryVerticesFromCosmicTracks', '*', 'No documentation'] ,
    '3':['nuclearInteractionMaker', '*', 'No documentation'] 
}

aod_title = "RecoVertex collections (in AOD only)"

aod = {
    '0':['offlinePrimaryVertices', 'reco::VertexCollection', 'Primary vertex reconstructed using the tracks taken from the generalTracks collection'],
    '1':['offlinePrimaryVerticesWithBS', '*', 'No documentation'] ,
    '2':['offlinePrimaryVerticesFromCosmicTracks', '*', 'No documentation'] ,
    '3':['nuclearInteractionMaker', '*', 'No documentation'] 
}
