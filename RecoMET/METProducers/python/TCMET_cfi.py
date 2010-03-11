import FWCore.ParameterSet.Config as cms

# File: TCMET.cff
# Author: R. Remington & F. Golf 
# Date: 11.14.2008
#
# Form Track Corrected MET

tcMet = cms.EDProducer("METProducer",
                       src = cms.InputTag("towerMaker"), #This parameter does not get used for TCMET
                       METType = cms.string('TCMET'),
                       alias = cms.string('TCMET'),
                       noHF = cms.bool(False),
                       globalThreshold = cms.double(0.0),
                       InputType = cms.string('CaloMET:Electron:Muon:Track'),  #This parameter does not get used for TCMET
                       electronInputTag  = cms.InputTag("gsfElectrons"),
                       muonInputTag      = cms.InputTag("muons"),
                       trackInputTag     = cms.InputTag("generalTracks"),
                       metInputTag       = cms.InputTag("met"),
                       beamSpotInputTag  = cms.InputTag("offlineBeamSpot"),
                       vertexInputTag    = cms.InputTag("offlinePrimaryVertices"),
                       muonDepValueMap   = cms.InputTag("muonMETValueMapProducer"  , "muCorrData"),     
                       tcmetDepValueMap  = cms.InputTag("muonTCMETValueMapProducer", "muCorrData"), 
                       pt_min  = cms.double(1.0),
                       pt_max  = cms.double(100.),
                       eta_max = cms.double(2.65), 
                       chi2_max = cms.double(5),
                       nhits_min = cms.double(6),
                       ptErr_max = cms.double(0.2),
                       track_quality = cms.vint32(2),
                       track_algos = cms.vint32(), 
                       isCosmics = cms.bool(False),
                       rf_type = cms.int32(1),
                       correctShowerTracks = cms.bool(False),
                       scaleShowerRF = cms.double(0.9),
                       scaleNoShowerRF = cms.double(1.02), 
                       usePvtxd0 = cms.bool(False),
                       nMinOuterHits = cms.int32(2),
                       usedeltaRRejection = cms.bool(False),
                       deltaRShower = cms.double(0.01),
                       checkTrackPropagation = cms.bool(False),
                       radius  = cms.double(130.),
                       zdist  = cms.double(314.),
                       corner = cms.double(1.479),
                       d0cuta = cms.double(0.1),
                       d0cutb = cms.double(0.) 
                       )



