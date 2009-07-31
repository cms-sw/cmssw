import FWCore.ParameterSet.Config as cms

TrackJetParameters = cms.PSet(
    src            = cms.InputTag("tracksForJets"),
    srcPVs         = cms.InputTag(''),
    jetType        = cms.string('BasicJet'),
    jetPtMin       = cms.double(1.0),
    inputEMin      = cms.double(0.0),
    inputEtMin     = cms.double(0.0),
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
      Ghost_EtaMax = cms.double(6.0)
    )

