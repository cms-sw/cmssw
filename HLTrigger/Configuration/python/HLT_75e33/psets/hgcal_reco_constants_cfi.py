import FWCore.ParameterSet.Config as cms

HGCAL_reco_constants = cms.PSet(
    dEdXweights = cms.vdouble(
        0.0, 9.205, 11.129999999999999, 11.129999999999999, 11.129999999999999,
        11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999,
        11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999,
        11.129999999999999, 11.129999999999999, 11.129999999999999, 13.2, 13.2,
        13.2, 13.2, 13.2, 13.2, 13.2,
        13.2, 35.745000000000005, 59.665000000000006, 60.7, 60.7,
        60.7, 60.7, 60.7, 60.7, 60.7,
        60.7, 60.7, 71.89, 83.08, 83.255,
        83.52000000000001, 83.61, 83.61, 83.61, 83.61,
        83.61, 83.61, 83.61
    ),
    fcPerEle = cms.double(0.00016020506),
    fcPerMip = cms.vdouble(
            2.06, 3.43, 5.15, 2.06, 3.43,
            5.15
        ),
    noises = cms.vdouble(
            2000.0, 2400.0, 2000.0, 2000.0, 2400.0,
            2000.0
        ),
    thicknessCorrection = cms.vdouble(
            0.75, 0.76, 0.75, 0.85, 0.85,
            0.84
        ),
    thresholdW0 = cms.vdouble(2.9, 2.9, 2.9),
    sciThicknessCorrection = cms.double(0.69),
    positionDeltaRho2 = cms.double(1.69),
    maxNumberOfThickIndices = cms.uint32(6),
    noiseMip = cms.PSet(
      scaleByDose = cms.bool(False),
      scaleByDoseAlgo = cms.uint32(0),
      scaleByDoseFactor = cms.double(1),
      referenceIdark = cms.double(-1),
      referenceXtalk = cms.double(-1),
      noise_MIP = cms.double(0.01)
    ),

)