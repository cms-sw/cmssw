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
    electronInputTag  = cms.InputTag("pixelMatchGsfElectrons"),
    muonInputTag      = cms.InputTag("muons"),
    trackInputTag     = cms.InputTag("generalTracks"),
    metInputTag       = cms.InputTag("met"),
    beamSpotInputTag  = cms.InputTag("offlineBeamSpot"),
    muonDepValueMap   = cms.InputTag("muonMETValueMapProducer"  , "muCorrData"),     
    tcmetDepValueMap  = cms.InputTag("muonTCMETValueMapProducer", "muCorrData"), 
    pt_min  = cms.double(2.0),
    pt_max  = cms.double(100.),
    eta_max = cms.double(2.4), 
    chi2_max = cms.double(4),
    nhits_min = cms.double(11),
    d0_max = cms.double(0.1)      
)



