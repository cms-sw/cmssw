import FWCore.ParameterSet.Config as cms

CaloJetParameters = cms.PSet(
    src            = cms.InputTag('towerMaker'),
    srcPVs         = cms.InputTag('offlinePrimaryVertices'),
    jetType        = cms.string('CaloJet'),
    # minimum jet pt
    jetPtMin       = cms.double(1.0),
    # minimum calo tower input et
    inputEtMin     = cms.double(0.5),
    # minimum calo tower input energy
    inputEMin      = cms.double(0.0),
    # primary vertex correction
    doPVCorrection = cms.bool(True),
    # pileup
    doPUOffsetCorr = cms.bool(False),
    doPUFastjet    = cms.bool(False),
      # if doPU is false, these are not read:
      Active_Area_Repeats = cms.int32(5),
      GhostArea = cms.double(0.01),
      Ghost_EtaMax = cms.double(6.0)
    )
