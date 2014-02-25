'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoPixelVertexing collections (in RECO and AOD)",
    "data": [
     {
      "instance": "pixelVertices",
      "container": "reco::VertexCollection",
      "desc": "primary vertices reconstructed from pixel tracks"
     },
     {
      "instance": "pixelTracks",
      "container": "reco::TrackCollection",
      "desc": "(proto)tracks created from two or three hits in Pixel detector"
     }
    ]
  },
  "aod": {
    "title": "RecoPixelVertexing collections (in AOD only)",
    "data": [

    ]
  },
  "reco": {
    "title": "RecoPixelVertexing collections (in RECO only)",
    "data": [
     {
      "instance": "pixelTracks",
      "container": "reco::TrackExtraCollection",
      "desc": "No documentation"
     }
    ]
  }
}
