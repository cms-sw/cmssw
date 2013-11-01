'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoVertex collections (in RECO and AOD)",
    "data": [
     {
      "instance": "offlinePrimaryVerticesWithBS",
      "container": "reco::VertexCollection ",
      "desc": "Primary vertex reconstructed using the tracks taken from the generalTracks collection, and imposing the offline beam spot as a constraint in the fit of the vertex position."
     },
     {
      "instance": "offlinePrimaryVertices",
      "container": "reco::VertexCollection ",
      "desc": "Primary vertex reconstructed using the tracks taken from the generalTracks collection"
     },
     {
      "instance": "nuclearInteractionMaker",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "offlinePrimaryVerticesFromCosmicTracks",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "generalV0Candidates:Lambda",
      "container": "reco::VertexCompositeCandidateCollection ",
      "desc": "Lambda0 candidate collection reconstructed using the tracks taken from the generalTracks collection"
     },
     {
      "instance": "generalV0Candidates:Kshort",
      "container": "reco::VertexCompositeCandidateCollection ",
      "desc": "K0S candidate collection reconstructed using the tracks taken from the generalTracks collection"
     },
     {
      "instance": "offlinePrimaryVerticesFromCTFTracks",
      "container": "reco::VertexCollection",
      "desc": "Primary vertex reconstructed using the CKF tracks, taken from the ctfWithMaterialTracks collection"
     },
     {
      "instance": "offlinePrimaryVerticesFromCTFTracks",
      "container": "reco::VertexCollection",
      "desc": "Identical to the offlinePrimaryVertices collection."
     },
     {
      "instance": "offlinePrimaryVerticesFromRSTracks",
      "container": "reco::VertexCollection",
      "desc": "Primary vertex reconstructed using the CKF tracks, taken from the rsWithMaterialTracks collection"
     }
    ]
  },
  "aod": {
    "title": "RecoVertex collections (in AOD only)",
    "data": [
     {
      "instance": "offlinePrimaryVerticesWithBS",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "offlinePrimaryVertices",
      "container": "reco::VertexCollection",
      "desc": "Primary vertex reconstructed using the tracks taken from the generalTracks collection"
     },
     {
      "instance": "nuclearInteractionMaker",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "offlinePrimaryVerticesFromCosmicTracks",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoVertex collections (in RECO only)",
    "data": [
     {
      "instance": "offlinePrimaryVerticesWithBS",
      "container": "reco::VertexCollection ",
      "desc": "Primary vertex reconstructed using the tracks taken from the generalTracks collection, and imposing the offline beam spot as a constraint in the fit of the vertex position."
     },
     {
      "instance": "offlinePrimaryVertices",
      "container": "reco::VertexCollection",
      "desc": "Primary vertex reconstructed using the tracks taken from the generalTracks collection"
     },
     {
      "instance": "nuclearInteractionMaker",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "offlinePrimaryVerticesFromCosmicTracks",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  }
}
