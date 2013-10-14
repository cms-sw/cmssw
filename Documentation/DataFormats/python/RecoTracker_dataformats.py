'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoTracker collections (in RECO and AOD)",
    "data": [
     {
      "instance": "dedxHarmonic2",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackExtrapolator",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "generalTracks",
      "container": "recoTrackExtras",
      "desc": "Track extra for the generalTracks.The trajectory state at the inner and outer most measurements"
     },
     {
      "instance": "generalTracks",
      "container": "recoTracks",
      "desc": "Collection of tracks obtained with tracker-standalone reconstruction and officially supported by the Tracker DPG group. Such a collection can contain tracks from different tracking algorithms"
     },
     {
      "instance": "extraFromSeeds",
      "container": "uints",
      "desc": "No documentation"
     },
     {
      "instance": "extraFromSeeds",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "beamhaloTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "generalTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "beamhaloTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "beamhaloTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "regionalCosmicTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "regionalCosmicTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "rsWithMaterialTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "regionalCosmicTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "rsWithMaterialTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "rsWithMaterialTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "conversionStepTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "conversionStepTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "ctfPixelLess",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "conversionStepTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "dedxDiscrimASmi",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "dedxTruncated40",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "RecoTracker collections (in AOD only)",
    "data": [
     {
      "instance": "rsWithMaterialTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "generalTracks",
      "container": "recoTracks",
      "desc": "Collection of tracks obtained with tracker-standalone reconstruction and officially supported by the Tracker DPG group. Such a collection can contain tracks from different tracking algorithms"
     },
     {
      "instance": "beamhaloTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "conversionStepTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "ctfPixelLess",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "regionalCosmicTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "dedxDiscrimASmi",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "dedxHarmonic2",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackExtrapolator",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoTracker collections (in RECO only)",
    "data": [
     {
      "instance": "dedxHarmonic2",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackExtrapolator",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "generalTracks",
      "container": "recoTrackExtras",
      "desc": "Track extra for the generalTracks.The trajectory state at the inner and outer most measurements"
     },
     {
      "instance": "generalTracks",
      "container": "recoTracks",
      "desc": "Collection of tracks obtained with tracker-standalone reconstruction and officially supported by the Tracker DPG group. Such a collection can contain tracks from different tracking algorithms"
     },
     {
      "instance": "extraFromSeeds",
      "container": "uints",
      "desc": "No documentation"
     },
     {
      "instance": "extraFromSeeds",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "beamhaloTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "generalTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "beamhaloTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "beamhaloTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "regionalCosmicTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "regionalCosmicTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "rsWithMaterialTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "regionalCosmicTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "rsWithMaterialTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "rsWithMaterialTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "conversionStepTracks",
      "container": "recoTrackExtras",
      "desc": "No documentation"
     },
     {
      "instance": "conversionStepTracks",
      "container": "recoTracks",
      "desc": "No documentation"
     },
     {
      "instance": "ctfPixelLess",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "conversionStepTracks",
      "container": "TrackingRecHitsOwned",
      "desc": "No documentation"
     },
     {
      "instance": "dedxDiscrimASmi",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "dedxTruncated40",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  }
}
