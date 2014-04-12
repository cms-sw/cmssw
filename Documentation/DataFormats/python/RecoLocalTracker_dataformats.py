'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoLocalTracker collections (in RECO and AOD)",
    "data": [
     {
      "instance": "siPixelDigis",
      "container": "DetIdedmEDCollection",
      "desc": "No documentation"
     },
     {
      "instance": "siStripDigis",
      "container": "DetIdedmEDCollection",
      "desc": "No documentation"
     },
     {
      "instance": "siStripClusters",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "siPixelClusters",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "RecoLocalTracker collections (in AOD only)",
    "data": [

    ]
  },
  "reco": {
    "title": "RecoLocalTracker collections (in RECO only)",
    "data": [
     {
      "instance": "siPixelDigis",
      "container": "DetIdedmEDCollection",
      "desc": "No documentation"
     },
     {
      "instance": "siStripDigis",
      "container": "DetIdedmEDCollection",
      "desc": "No documentation"
     },
     {
      "instance": "siStripClusters",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "siPixelClusters",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  }
}
