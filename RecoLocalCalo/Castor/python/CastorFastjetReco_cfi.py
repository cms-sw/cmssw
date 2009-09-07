import FWCore.ParameterSet.Config as cms

#from RecoJets.JetProducers.CaloJetParameters_cfi import *
#from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

CaloJetParameters = cms.PSet(
    module_label = cms.string('CastorFastjetReco'),
    src            = cms.InputTag('CastorTowerCandidateReco'),
    srcPVs         = cms.InputTag(''),
    jetType        = cms.string('BasicJet'),
    # minimum jet pt
    jetPtMin       = cms.double(0.0),
    # minimum calo tower input et
    inputEtMin     = cms.double(0.0),
    # minimum calo tower input energy
    inputEMin      = cms.double(0.0),
    # primary vertex correction
    doPVCorrection = cms.bool(False),
    # pileup with offset correction
    doPUOffsetCorr = cms.bool(False),
       # if pileup is false, these are not read:
       nSigmaPU = cms.double(1.0),
       radiusPU = cms.double(0.5),  
    # fastjet-style pileup 
    doPUFastjet    = cms.bool(False),
       # if doPU is false, these are not read:
       Active_Area_Repeats = cms.int32(5),
       GhostArea = cms.double(0.01),
       Ghost_EtaMax = cms.double(6.0),
    )
    
GenJetParameters = cms.PSet(
    src            = cms.InputTag('genParticlesForJets'),
    srcPVs         = cms.InputTag(''),
    jetType        = cms.string('GenJet'),
    # minimum jet pt
    jetPtMin       = cms.double(0.0),
    # minimum calo tower input et
    inputEtMin     = cms.double(0.0),
    # minimum calo tower input energy
    inputEMin      = cms.double(0.0),
    # primary vertex correction
    doPVCorrection = cms.bool(False),
    # pileup with offset correction
    doPUOffsetCorr = cms.bool(False),
       # if pileup is false, these are not read:
       nSigmaPU = cms.double(1.0),
       radiusPU = cms.double(0.5),  
    # fastjet-style pileup 
    doPUFastjet    = cms.bool(False),
       # if doPU is false, these are not read:
       Active_Area_Repeats = cms.int32(5),
       GhostArea = cms.double(0.01),
       Ghost_EtaMax = cms.double(6.0),
    )
        
AnomalousCellParameters = cms.PSet(
    maxBadEcalCells         = cms.uint32(9999999),
    maxRecoveredEcalCells   = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxBadHcalCells         = cms.uint32(9999999),
    maxRecoveredHcalCells   = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999)
    )



CastorFastjetRecoKt = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(1.0)
    )
    
CastorFastjetRecoSISCone = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("SISCone"),
    rParam       = cms.double(1.0)
    )
    
CastorFastjetRecoSISConeGen = cms.EDProducer(
    "FastjetJetProducer",
    GenJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("SISCone"),
    rParam       = cms.double(1.0)
    )
    
CastorFastjetRecoKtGen = cms.EDProducer(
    "FastjetJetProducer",
    GenJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(1.0)
    )
