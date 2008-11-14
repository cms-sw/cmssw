import FWCore.ParameterSet.Config as cms

# File: TCMET.cff
# Author: R. Remington & F. Golf 
# Date: 11.14.2008
#
# Form Track Corrected MET

tcMet = cms.EDProducer(
    "METProducer",
    src = cms.InputTag("towerMaker"), #This parameter does not get used for TCMET
    METType = cms.string('MET'),
    alias = cms.string('TCMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CaloMET:Electron:Muon:Track'),  #This parameter does not get used for TCMET  
    electronLabel = cms.InputTag("pixelMatchGsfElectrons"),
    muonLabel     = cms.InputTag("muons"),
    trackLabel    = cms.InputTag("tracks"),
    metLabel      = cms.InputTag("met")
    )



