import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

ak7BasicJets = cms.EDProducer(
    "FastjetJetProducer",
    AnomalousCellParameters,
    src            = cms.InputTag('CastorTowerReco'),
    srcPVs         = cms.InputTag('offlinePrimaryVertices'),
    jetType        = cms.string('BasicJet'),
    # minimum jet pt
    jetPtMin       = cms.double(0.0),
    # minimum calo tower input et
    inputEtMin     = cms.double(0.0),
    # minimum calo tower input energy
    inputEMin      = cms.double(0.0),
    # primary vertex correction
    doPVCorrection = cms.bool(True),
    # pileup with offset correction
    doPUOffsetCorr = cms.bool(False),
       # if pileup is false, these are not read:
       nSigmaPU = cms.double(1.0),
       radiusPU = cms.double(0.5),  
    # fastjet-style pileup 
    doAreaFastjet    = cms.bool(False),
    doRhoFastjet     = cms.bool(False),
       # if doPU is false, these are not read:
       Active_Area_Repeats = cms.int32(1),
       GhostArea = cms.double(0.01),
       Ghost_EtaMax = cms.double(5.0),
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.7)
    )

