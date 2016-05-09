import FWCore.ParameterSet.Config as cms

trackingRecHitProducer = cms.EDProducer("TrackingRecHitProducer",
     simHits = cms.InputTag("famosSimHits","TrackerHits"),
     ############ Matthias's Example############
     plugins=cms.VPSet(
         cms.PSet(
             name = cms.string("smearPixels"),
             type=cms.string("TrackingRecHitNoSmearingPlugin"),
             select=cms.string("subdetId==BPX || subdetId==FPX"),
             errorXX = cms.double(0.05*0.05),
             errorYY = cms.double(0.05*0.05),
             errorXY = cms.double(0.0),
         ),
         cms.PSet(
             name = cms.string("smearStrips"),
             type=cms.string("TrackingRecHitNoSmearingPlugin"),
             select=cms.string("subdetId==TIB || subdetId==TID ||subdetId==TOB || subdetId==TEC"),
             errorXX = cms.double(0.1*0.1),
             errorYY = cms.double(2*2),
             errorXY = cms.double(0.0),
         )
     )
)

trackingRecHitProducer_alt = cms.EDProducer("TrackingRecHitProducer",
     simHits = cms.InputTag("famosSimHits","TrackerHits"),
     ############ Matthias's Example############
     plugins=cms.VPSet(
         cms.PSet(
             name = cms.string("smearPixels"),
             type=cms.string("TrackingRecHitNoSmearingPlugin"),
             select=cms.string("subdetId==BPX || subdetId==FPX"),
             errorXX = cms.double(0.05*0.05),
             errorYY = cms.double(0.05*0.05),
             errorXY = cms.double(0.0),
         )
     )
)

trackerStripGaussianResolutions={
    "TIB": {
        1: cms.double(0.00195),
        2: cms.double(0.00191),
        3: cms.double(0.00325),
        4: cms.double(0.00323)
    },
    "TID": {
        1: cms.double(0.00262),
        2: cms.double(0.00354),
        3: cms.double(0.00391)
    },
    "TOB": {
        1: cms.double(0.00461),
        2: cms.double(0.00458),
        3: cms.double(0.00488),
        4: cms.double(0.00491),
        5: cms.double(0.00293),
        6: cms.double(0.00299)
    },
    "TEC": {
        1: cms.double(0.00262),
        2: cms.double(0.00354),
        3: cms.double(0.00391),
        4: cms.double(0.00346),
        5: cms.double(0.00378),
        6: cms.double(0.00508),
        7: cms.double(0.00422),
        8: cms.double(0.00508), # give me proper values
        9: cms.double(0.00422), # give me proper values
    }  
}

for subdetId,trackerLayers in trackerStripGaussianResolutions.iteritems():
    for trackerLayer, resolutionX in trackerLayers.iteritems():
        pluginConfig = cms.PSet(
            name = cms.string(subdetId+str(trackerLayer)),
            type=cms.string("TrackingRecHitStripGSSmearingPlugin"),
            resolutionX=cms.double(0.01),
            select=cms.string("(subdetId=="+subdetId+") && (layer=="+str(trackerLayer)+")"),
        )
        trackingRecHitProducer_alt.plugins.append(pluginConfig)

