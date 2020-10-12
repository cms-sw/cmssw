import FWCore.ParameterSet.Config as cms

tkEgConfig = cms.PSet(
    debug=cms.untracked.int32(0),
    doBremRecovery=cms.bool(True),
    filterHwQuality=cms.bool(True),
    caloHwQual=cms.int32(4),
    dEtaMaxBrem=cms.double(0.02),
    dPhiMaxBrem=cms.double(0.1),
    absEtaBoundaries=cms.vdouble(0.0, 0.9, 1.5),
    dEtaValues=cms.vdouble(0.025, 0.015, 0.01),  # last was  0.0075  in TDR
    dPhiValues=cms.vdouble(0.07, 0.07, 0.07),
    caloEtMin=cms.double(0.0),
    trkQualityPtMin=cms.double(10.0),
    trkQualityChi2=cms.double(1e10),
    writeEgSta=cms.bool(True)
)
