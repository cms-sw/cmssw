import FWCore.ParameterSet.Config as cms

### Apparently we need to make HCAL isolation from Towers ourselves
from RecoEgamma.EgammaIsolationAlgos.gamHcalExtractorBlocks_cff import *
gamIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",   
     src = cms.InputTag("photons"),      
     MultipleDepositsFlag = cms.bool(False),     
     trackType = cms.string('candidate'),    
     ExtractorPSet = cms.PSet( GamIsoHcalFromTowersExtractorBlock )      
)   


# definee   module labels for POG isolation
patAODPhotonIsolationLabels = cms.VInputTag(
        cms.InputTag("gamIsoDepositTk"),
        cms.InputTag("gamIsoDepositEcalFromHits"),
        cms.InputTag("gamIsoDepositHcalFromTowers"),     
)

# read and convert to ValueMap<IsoDeposit> keyed to Candidate
patAODPhotonIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    collection   = cms.InputTag("photons"),
    associations =  patAODPhotonIsolationLabels,
)

# re-key ValueMap<IsoDeposit> to Layer 0 output
layer0PhotonIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    collection   = cms.InputTag("allLayer0Photons"),
    backrefs     = cms.InputTag("allLayer0Photons"),
    commonLabel  = cms.InputTag("patAODPhotonIsolations"),
    associations =  patAODPhotonIsolationLabels,
)

# sequence to run on AOD before PAT 
patAODPhotonIsolation = cms.Sequence(gamIsoDepositHcalFromTowers * patAODPhotonIsolations)

# sequence to run after PAT cleaners
patLayer0PhotonIsolation = cms.Sequence(layer0PhotonIsolations)

