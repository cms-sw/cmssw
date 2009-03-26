import FWCore.ParameterSet.Config as cms

### OK, so apparently we need to make IsoDeposits for HCAL Towers ourselves in 2.2.X
from RecoEgamma.EgammaIsolationAlgos.eleHcalExtractorBlocks_cff import *
eleIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",      
        src = cms.InputTag("pixelMatchGsfElectrons"),   
        MultipleDepositsFlag = cms.bool(False),     
        trackType = cms.string('candidate'),    
        ExtractorPSet = cms.PSet( EleIsoHcalFromTowersExtractorBlock )      
)

### Compute isolation values, using POG modules
from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff  import *

# sequence to run on AOD 
patElectronIsolation = cms.Sequence(
    eleIsoFromDepsTk +
    eleIsoFromDepsEcalFromHits +
    eleIsoDepositHcalFromTowers * eleIsoFromDepsHcalFromTowers 
)

