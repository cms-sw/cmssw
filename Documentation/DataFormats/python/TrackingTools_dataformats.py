'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "TrackingTools collections (in RECO and AOD)",
    "data": [
     {
      "instance": "GsfGlobalElectronTest",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "CkfElectronCandidates",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "electronGsfTracks",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "electronMergedSeeds",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "TrackingTools collections (in AOD only)",
    "data": [
     {
      "instance": "electronGsfTracks",
      "container": "recoGsfTracks",
      "desc": "No documentation"
     },
     {
      "instance": "GsfGlobalElectronTest",
      "container": "recoTracks",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "TrackingTools collections (in RECO only)",
    "data": [
     {
      "instance": "GsfGlobalElectronTest",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "CkfElectronCandidates",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "electronGsfTracks",
      "container": "recoGsfTracks",
      "desc": "No documentation"
     },
     {
      "instance": "electronMergedSeeds",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "electronGsfTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "electronGsfTracks",
      "container": "recoGsfTrackExtras",
      "desc": "No documentation"
     }
    ]
  }
}
