import FWCore.ParameterSet.Config as cms

### Apparently we need to make HCAL isolation from Towers ourselves
from RecoEgamma.EgammaIsolationAlgos.gamHcalExtractorBlocks_cff import *
gamIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",   
     src = cms.InputTag("photons"),      
     MultipleDepositsFlag = cms.bool(False),     
     trackType = cms.string('candidate'),    
     ExtractorPSet = cms.PSet( GamIsoHcalFromTowersExtractorBlock )      
)   

### Compute isolation values, using POG modules
from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff  import *

# sequence to run on AOD 
patPhotonIsolation = cms.Sequence(
    gamIsoFromDepsTk +
    gamIsoFromDepsEcalFromHits +
    gamIsoDepositHcalFromTowers * gamIsoFromDepsHcalFromTowers 
)
