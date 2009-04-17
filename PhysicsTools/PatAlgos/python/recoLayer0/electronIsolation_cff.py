import FWCore.ParameterSet.Config as cms

### OK, so apparently we need to make IsoDeposits for HCAL Towers ourselves in 2.2.X
from RecoEgamma.EgammaIsolationAlgos.eleHcalExtractorBlocks_cff import *
eleIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",      
        src = cms.InputTag("pixelMatchGsfElectrons"),   
        MultipleDepositsFlag = cms.bool(False),     
        trackType = cms.string('candidate'),    
        ExtractorPSet = cms.PSet( EleIsoHcalFromTowersExtractorBlock )      
)

# define module labels for old (tk-based isodeposit) POG isolation
# WARNING: these labels are used in the functions below.
patAODElectronIsolationLabels = cms.VInputTag(
        cms.InputTag("eleIsoDepositTk"),
        cms.InputTag("eleIsoDepositEcalFromHits"),
        cms.InputTag("eleIsoDepositHcalFromTowers"),
)

# read and convert to ValueMap<IsoDeposit> keyed to Candidate
patAODElectronIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    collection   = cms.InputTag("pixelMatchGsfElectrons"),
    associations = patAODElectronIsolationLabels,
)

# re-key ValueMap<IsoDeposit> to Layer 0 output
layer0ElectronIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    collection   = cms.InputTag("allLayer0Electrons"),
    backrefs     = cms.InputTag("allLayer0Electrons"),
    commonLabel  = cms.InputTag("patAODElectronIsolations"),
    associations = patAODElectronIsolationLabels,
)

# sequence to run on AOD before PAT
patAODElectronIsolation = cms.Sequence(eleIsoDepositHcalFromTowers * patAODElectronIsolations)

# sequence to run after the PAT cleaners
patLayer0ElectronIsolation = cms.Sequence(layer0ElectronIsolations)
