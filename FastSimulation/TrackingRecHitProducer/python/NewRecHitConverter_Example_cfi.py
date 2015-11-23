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
