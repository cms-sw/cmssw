import FWCore.ParameterSet.Config as cms

l1tTowerCalibration = cms.EDProducer("L1TowerCalibrator",
    EcalTpEtMin = cms.double(0.5),
    HFTpEtMin = cms.double(0.5),
    HGCalEmTpEtMin = cms.double(0.25),
    HGCalHadTpEtMin = cms.double(0.25),
    HcalTpEtMin = cms.double(0.5),
    L1HgcalTowersInputTag = cms.InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor"),
    barrelSF = cms.double(1.0),
    debug = cms.bool(False),
    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    hfSF = cms.double(1.0),
    hgcalSF = cms.double(1.0),
    l1CaloTowers = cms.InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection"),
    nHits_to_nvtx_params = cms.VPSet(
        cms.PSet(
            fit = cms.string('hf'),
            params = cms.vdouble(165.706, 0.153)
        ),
        cms.PSet(
            fit = cms.string('ecal'),
            params = cms.vdouble(168.055, 0.377)
        ),
        cms.PSet(
            fit = cms.string('hgcalEM'),
            params = cms.vdouble(157.522, 0.09)
        ),
        cms.PSet(
            fit = cms.string('hgcalHad'),
            params = cms.vdouble(159.295, 0.178)
        ),
        cms.PSet(
            fit = cms.string('hcal'),
            params = cms.vdouble(168.548, 0.362)
        )
    ),
    nvtx_to_PU_sub_params = cms.VPSet(
        cms.PSet(
            calo = cms.string('ecal'),
            iEta = cms.string('er1to3'),
            params = cms.vdouble(0.008955, 0.000298)
        ),
        cms.PSet(
            calo = cms.string('ecal'),
            iEta = cms.string('er4to6'),
            params = cms.vdouble(0.009463, 0.000256)
        ),
        cms.PSet(
            calo = cms.string('ecal'),
            iEta = cms.string('er7to9'),
            params = cms.vdouble(0.008783, 0.000293)
        ),
        cms.PSet(
            calo = cms.string('ecal'),
            iEta = cms.string('er10to12'),
            params = cms.vdouble(0.009308, 0.000255)
        ),
        cms.PSet(
            calo = cms.string('ecal'),
            iEta = cms.string('er13to15'),
            params = cms.vdouble(0.00929, 0.000221)
        ),
        cms.PSet(
            calo = cms.string('ecal'),
            iEta = cms.string('er16to18'),
            params = cms.vdouble(0.009282, 0.000135)
        ),
        cms.PSet(
            calo = cms.string('hcal'),
            iEta = cms.string('er1to3'),
            params = cms.vdouble(0.009976, 0.000377)
        ),
        cms.PSet(
            calo = cms.string('hcal'),
            iEta = cms.string('er4to6'),
            params = cms.vdouble(0.009803, 0.000394)
        ),
        cms.PSet(
            calo = cms.string('hcal'),
            iEta = cms.string('er7to9'),
            params = cms.vdouble(0.009654, 0.000429)
        ),
        cms.PSet(
            calo = cms.string('hcal'),
            iEta = cms.string('er10to12'),
            params = cms.vdouble(0.009107, 0.000528)
        ),
        cms.PSet(
            calo = cms.string('hcal'),
            iEta = cms.string('er13to15'),
            params = cms.vdouble(0.008367, 0.000652)
        ),
        cms.PSet(
            calo = cms.string('hcal'),
            iEta = cms.string('er16to18'),
            params = cms.vdouble(0.009681, 9.6e-05)
        ),
        cms.PSet(
            calo = cms.string('hgcalEM'),
            iEta = cms.string('er1p4to1p8'),
            params = cms.vdouble(-0.011772, 0.004142)
        ),
        cms.PSet(
            calo = cms.string('hgcalEM'),
            iEta = cms.string('er1p8to2p1'),
            params = cms.vdouble(-0.015488, 0.00541)
        ),
        cms.PSet(
            calo = cms.string('hgcalEM'),
            iEta = cms.string('er2p1to2p4'),
            params = cms.vdouble(-0.02115, 0.006078)
        ),
        cms.PSet(
            calo = cms.string('hgcalEM'),
            iEta = cms.string('er2p4to2p7'),
            params = cms.vdouble(-0.015705, 0.005339)
        ),
        cms.PSet(
            calo = cms.string('hgcalEM'),
            iEta = cms.string('er2p7to3p1'),
            params = cms.vdouble(-0.018492, 0.00562)
        ),
        cms.PSet(
            calo = cms.string('hgcalHad'),
            iEta = cms.string('er1p4to1p8'),
            params = cms.vdouble(0.005675, 0.000615)
        ),
        cms.PSet(
            calo = cms.string('hgcalHad'),
            iEta = cms.string('er1p8to2p1'),
            params = cms.vdouble(0.00456, 0.001099)
        ),
        cms.PSet(
            calo = cms.string('hgcalHad'),
            iEta = cms.string('er2p1to2p4'),
            params = cms.vdouble(3.6e-05, 0.001608)
        ),
        cms.PSet(
            calo = cms.string('hgcalHad'),
            iEta = cms.string('er2p4to2p7'),
            params = cms.vdouble(0.000869, 0.001754)
        ),
        cms.PSet(
            calo = cms.string('hgcalHad'),
            iEta = cms.string('er2p7to3p1'),
            params = cms.vdouble(-0.006574, 0.003134)
        ),
        cms.PSet(
            calo = cms.string('hf'),
            iEta = cms.string('er29to33'),
            params = cms.vdouble(-0.203291, 0.044096)
        ),
        cms.PSet(
            calo = cms.string('hf'),
            iEta = cms.string('er34to37'),
            params = cms.vdouble(-0.210922, 0.045628)
        ),
        cms.PSet(
            calo = cms.string('hf'),
            iEta = cms.string('er38to41'),
            params = cms.vdouble(-0.229562, 0.05056)
        )
    ),
    puThreshold = cms.double(5.0),
    puThresholdEcal = cms.double(2.0),
    puThresholdEcalMax = cms.double(1.5),
    puThresholdEcalMin = cms.double(0.75),
    puThresholdHFMax = cms.double(15.0),
    puThresholdHFMin = cms.double(10.0),
    puThresholdHGCalEMMax = cms.double(1.75),
    puThresholdHGCalEMMin = cms.double(1.25),
    puThresholdHGCalHadMax = cms.double(1.25),
    puThresholdHGCalHadMin = cms.double(0.75),
    puThresholdHcal = cms.double(3.0),
    puThresholdHcalMax = cms.double(2.0),
    puThresholdHcalMin = cms.double(1.0),
    puThresholdL1eg = cms.double(2.0),
    skipCalibrations = cms.bool(False)
)
