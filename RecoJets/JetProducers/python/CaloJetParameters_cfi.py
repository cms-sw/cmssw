import FWCore.ParameterSet.Config as cms

CaloJetParameters = cms.PSet(
    src            = cms.InputTag('towerMaker'),
    srcPVs         = cms.InputTag('offlinePrimaryVertices'),
    jetType        = cms.string('CaloJet'),
    # minimum jet pt
    jetPtMin       = cms.double(3.0),
    # minimum calo tower input et
    inputEtMin     = cms.double(0.3),
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
       Jet_EtaMax = cms.double(4.5)
    )
