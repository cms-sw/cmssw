import FWCore.ParameterSet.Config as cms

l1tHGCalVFEProducer = cms.EDProducer("HGCalVFEProducer",
    ProcessorParameters = cms.PSet(
        ProcessorName = cms.string('HGCalVFEProcessorSums'),
        calibrationCfg_ee = cms.PSet(
            chargeCollectionEfficiency = cms.PSet(
                refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
            ),
            dEdXweights = cms.vdouble(
                0, 9.205, 0, 22.259999999999998, 0,
                22.259999999999998, 0, 22.259999999999998, 0, 22.259999999999998,
                0, 22.259999999999998, 0, 22.259999999999998, 0,
                22.259999999999998, 0, 22.259999999999998, 0, 26.4,
                0, 26.4, 0, 26.4, 0,
                26.4, 0, 95.41000000000001, 60.7, 60.7,
                60.7, 60.7, 60.7, 60.7, 60.7,
                60.7, 60.7, 71.89, 83.08, 83.255,
                83.52000000000001, 83.61, 83.61, 83.61, 83.61,
                83.61, 83.61, 83.61
            ),
            fCperMIP = cms.vdouble(2.06, 3.43, 5.15, 3.43),
            lsb = cms.double(0.09765625),
            thicknessCorrection = cms.vdouble(0.75, 0.76, 0.75, 0.76)
        ),
        calibrationCfg_hesc = cms.PSet(
            chargeCollectionEfficiency = cms.PSet(
                values = cms.vdouble(1.0)
            ),
            dEdXweights = cms.vdouble(
                0, 9.205, 0, 22.259999999999998, 0,
                22.259999999999998, 0, 22.259999999999998, 0, 22.259999999999998,
                0, 22.259999999999998, 0, 22.259999999999998, 0,
                22.259999999999998, 0, 22.259999999999998, 0, 26.4,
                0, 26.4, 0, 26.4, 0,
                26.4, 0, 95.41000000000001, 60.7, 60.7,
                60.7, 60.7, 60.7, 60.7, 60.7,
                60.7, 60.7, 71.89, 83.08, 83.255,
                83.52000000000001, 83.61, 83.61, 83.61, 83.61,
                83.61, 83.61, 83.61
            ),
            fCperMIP = cms.vdouble(1.0),
            lsb = cms.double(0.067138671875),
            thicknessCorrection = cms.vdouble(0.69)
        ),
        calibrationCfg_hesi = cms.PSet(
            chargeCollectionEfficiency = cms.PSet(
                refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
            ),
            dEdXweights = cms.vdouble(
                0, 9.205, 0, 22.259999999999998, 0,
                22.259999999999998, 0, 22.259999999999998, 0, 22.259999999999998,
                0, 22.259999999999998, 0, 22.259999999999998, 0,
                22.259999999999998, 0, 22.259999999999998, 0, 26.4,
                0, 26.4, 0, 26.4, 0,
                26.4, 0, 95.41000000000001, 60.7, 60.7,
                60.7, 60.7, 60.7, 60.7, 60.7,
                60.7, 60.7, 71.89, 83.08, 83.255,
                83.52000000000001, 83.61, 83.61, 83.61, 83.61,
                83.61, 83.61, 83.61
            ),
            fCperMIP = cms.vdouble(2.06, 3.43, 5.15, 3.43),
            lsb = cms.double(0.09765625),
            thicknessCorrection = cms.vdouble(0.85, 0.85, 0.84, 0.85)
        ),
        calibrationCfg_nose = cms.PSet(
            chargeCollectionEfficiency = cms.PSet(
                refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
            ),
            dEdXweights = cms.vdouble(
                0.0, 39.500245, 39.756638, 39.756638, 39.756638,
                39.756638, 66.020266, 92.283895, 92.283895
            ),
            fCperMIP = cms.vdouble(2.06, 3.43, 5.15, 3.43),
            lsb = cms.double(0.09765625),
            thicknessCorrection = cms.vdouble(1.132, 1.092, 1.084)
        ),
        compressionCfg_hdm = cms.PSet(
            exponentBits = cms.uint32(4),
            mantissaBits = cms.uint32(3),
            rounding = cms.bool(True),
            truncationBits = cms.uint32(3)
        ),
        compressionCfg_ldm = cms.PSet(
            exponentBits = cms.uint32(4),
            mantissaBits = cms.uint32(3),
            rounding = cms.bool(True),
            truncationBits = cms.uint32(1)
        ),
        linearizationCfg_sc = cms.PSet(
            adcnBits = cms.uint32(10),
            adcsaturation = cms.double(68.75),
            linLSB = cms.double(0.067138671875),
            linnBits = cms.uint32(17),
            oot_coefficients = cms.vdouble(0.0, 0.0),
            tdcOnset = cms.double(55),
            tdcnBits = cms.uint32(12),
            tdcsaturation = cms.double(1000)
        ),
        linearizationCfg_si = cms.PSet(
            adcnBits = cms.uint32(10),
            adcsaturation = cms.double(100),
            linLSB = cms.double(0.09765625),
            linnBits = cms.uint32(17),
            oot_coefficients = cms.vdouble(0.0, 0.0),
            tdcOnset = cms.double(60),
            tdcnBits = cms.uint32(12),
            tdcsaturation = cms.double(10000)
        ),
        summationCfg = cms.PSet(
            noiseScintillator = cms.PSet(
                refToPSet_ = cms.string('HGCAL_noise_heback')
            ),
            noiseSilicon = cms.PSet(
                refToPSet_ = cms.string('HGCAL_noise_fC')
            ),
            noiseThreshold = cms.double(3),
            scintillatorCellLSB_MIP = cms.double(0.067138671875),
            siliconCellLSB_fC = cms.double(0.09765625)
        )
    ),
    bhDigis = cms.InputTag("simHGCalUnsuppressedDigis","HEback"),
    eeDigis = cms.InputTag("simHGCalUnsuppressedDigis","EE"),
    fhDigis = cms.InputTag("simHGCalUnsuppressedDigis","HEfront")
)
