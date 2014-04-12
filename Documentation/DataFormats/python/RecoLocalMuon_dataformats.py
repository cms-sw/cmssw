'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoLocalMuon collections (in RECO and AOD)",
    "data": [
     {
      "instance": "dt4DSegments",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "dt1DRecHits",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "cscSegments",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "csc2DRecHits",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "RecoLocalMuon collections (in AOD only)",
    "data": [

    ]
  },
  "reco": {
    "title": "RecoLocalMuon collections (in RECO only)",
    "data": [
     {
      "instance": "dt4DSegments",
      "container": "DTRecSegment4DCollection",
      "desc": "DT segment in a full chamber"
     },
     {
      "instance": "dt1DRecHits",
      "container": "DTRecHitCollection",
      "desc": "DT 1D rechits with L/R ambiguity"
     },
     {
      "instance": "dt4DCosmicSegments",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "dt1DCosmicRecHits",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "cscSegments",
      "container": "CSCSegmentCollection",
      "desc": "CSC segments [A CSCSegment is built from the rechits in one CSC on a track; a CSCSegment is also, formally, itself a rechit]"
     },
     {
      "instance": "csc2DRecHits",
      "container": "CSCRecHit2DCollection",
      "desc": "CSC rechits [A CSCRecHit2D is a reconstructed hit on one layer of a CSC. Effectively 3-dim: 2-d x, y but CSCLayer - labelled by CSCDetId - gives z]"
     },
     {
      "instance": "RPCRecHit",
      "container": "RPCRecHitCollection",
      "desc": "RPC rechits"
     },
     {
      "instance": "DTSLRecSegment2D",
      "container": "DTRecSegment2DCollection",
      "desc": "DT segment in one projection"
     }
    ]
  }
}
