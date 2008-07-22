import FWCore.ParameterSet.Config as cms

# load E/gamma POG config
from RecoEgamma.EgammaIsolationAlgos.gamIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.gamEcalExtractorBlocks_cff import *
from RecoEgamma.EgammaIsolationAlgos.gamHcalExtractorBlocks_cff import *

gamIsoDepositTk.ExtractorPSet.ComponentName = cms.string('TrackExtractor')
gamIsoDepositEcalSCVetoFromClusts = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( GamIsoEcalSCVetoFromClustsExtractorBlock )
)
gamIsoDepositEcalFromClusts = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( GamIsoEcalFromClustsExtractorBlock )
)
gamIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( GamIsoHcalFromTowersExtractorBlock )
)

# define module labels for POG isolation
patAODPhotonIsolationLabels = cms.VInputTag(
        cms.InputTag("gamIsoDepositTk"),
     #   cms.InputTag("gamIsoDepositEcalFromHits"),
     #   cms.InputTag("gamIsoDepositHcalFromHits"),
       cms.InputTag("gamIsoDepositEcalFromClusts"),       # try these two if you want to compute them from AOD
       cms.InputTag("gamIsoDepositHcalFromTowers"),       # instead of reading the values computed in RECO
     #  cms.InputTag("gamIsoDepositEcalSCVetoFromClusts"), # this is an alternative to 'gamIsoDepositEcalFromClusts'
)

# read and convert to ValueMap<IsoDeposit> keyed to Candidate
patAODPhotonIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    collection   = cms.InputTag("photons"),
    associations =  patAODPhotonIsolationLabels,
)

# re-key to PAT Layer 0 output
layer0PhotonIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    collection   = cms.InputTag("allLayer0Photons"),
    backrefs     = cms.InputTag("allLayer0Photons"),
    commonLabel  = cms.InputTag("patAODPhotonIsolations"),
    associations =  patAODPhotonIsolationLabels,
)

# selecting POG modules that can run on top of AOD
gamIsoDepositAOD = cms.Sequence(gamIsoDepositTk * gamIsoDepositEcalFromClusts * gamIsoDepositHcalFromTowers)

# sequence to run on AOD before PAT 
patAODPhotonIsolation = cms.Sequence(gamIsoDepositAOD * patAODPhotonIsolations)
# patAODPhotonIsolation = cms.Sequence(patAODPhotonIsolations)

# sequence to run after PAT cleaners
patLayer0PhotonIsolation = cms.Sequence(layer0PhotonIsolations)

