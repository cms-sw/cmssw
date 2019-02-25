import FWCore.ParameterSet.Config as cms

L1TowerCalibrationProducer = cms.EDProducer("L1TowerCalibrator",
    # Choosen settings (v8 24 Jan 2019)
    HcalTpEtMin = cms.double(0.0),
    EcalTpEtMin = cms.double(0.0),
    HGCalHadTpEtMin = cms.double(0.25),
    HGCalEmTpEtMin = cms.double(0.25),
    HFTpEtMin = cms.double(0.5),
    puThreshold = cms.double(5.0),
    puThresholdEcal = cms.double(2.0),
    puThresholdHcal = cms.double(3.0),
    puThresholdL1eg = cms.double(4.0),
    puThresholdHGCalEMMin = cms.double(1.0),
    puThresholdHGCalEMMax = cms.double(1.5),
    puThresholdHGCalHadMin = cms.double(0.5),
    puThresholdHGCalHadMax = cms.double(1.0),
    puThresholdHFMin = cms.double(4.0),
    puThresholdHFMax = cms.double(10.0),
    barrelSF = cms.double(1.0),
    hgcalSF = cms.double(1.0),
    hfSF = cms.double(1.0),
    debug = cms.bool(False),
    skipCalibrations = cms.bool(False),
    #debug = cms.bool(True),
    l1CaloTowers = cms.InputTag("L1EGammaClusterEmuProducer","L1CaloTowerCollection"),
    L1HgcalTowersInputTag = cms.InputTag("hgcalTowerProducer:HGCalTowerProcessor"),
    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    nHits_to_nvtx_params = cms.VPSet( # Parameters derived on 27 Jan 2019
        cms.PSet(
            fit = cms.string( "hf" ),
            params = cms.vdouble( -0.695, 0.486 )
        ),
        cms.PSet(
            fit = cms.string( "ecal" ),
            params = cms.vdouble( -14.885, 0.666 )
        ),
        cms.PSet(
            fit = cms.string( "hgcalEM" ),
            params = cms.vdouble( -0.334, 0.278 )
        ),
        cms.PSet(
            fit = cms.string( "hgcalHad" ),
            params = cms.vdouble( -1.752, 0.485 )
        ),
        cms.PSet(
            fit = cms.string( "hcal" ),
            params = cms.vdouble( -11.713, 1.574 )
        ),
    ),
	nvtx_to_PU_sub_params = cms.VPSet(
		cms.PSet(
			calo = cms.string( "ecal" ),
			iEta = cms.string( "er1to3" ),
			params = cms.vdouble( 0.015630, 0.000701 )
		),
		cms.PSet(
			calo = cms.string( "ecal" ),
			iEta = cms.string( "er4to6" ),
			params = cms.vdouble( 0.010963, 0.000590 )
		),
		cms.PSet(
			calo = cms.string( "ecal" ),
			iEta = cms.string( "er7to9" ),
			params = cms.vdouble( 0.003597, 0.000593 )
		),
		cms.PSet(
			calo = cms.string( "ecal" ),
			iEta = cms.string( "er10to12" ),
			params = cms.vdouble( -0.000197, 0.000492 )
		),
		cms.PSet(
			calo = cms.string( "ecal" ),
			iEta = cms.string( "er13to15" ),
			params = cms.vdouble( -0.001255, 0.000410 )
		),
		cms.PSet(
			calo = cms.string( "ecal" ),
			iEta = cms.string( "er16to18" ),
			params = cms.vdouble( -0.001140, 0.000248 )
		),
		cms.PSet(
			calo = cms.string( "hcal" ),
			iEta = cms.string( "er1to3" ),
			params = cms.vdouble( -0.003391, 0.001630 )
		),
		cms.PSet(
			calo = cms.string( "hcal" ),
			iEta = cms.string( "er4to6" ),
			params = cms.vdouble( -0.004845, 0.001809 )
		),
		cms.PSet(
			calo = cms.string( "hcal" ),
			iEta = cms.string( "er7to9" ),
			params = cms.vdouble( -0.005202, 0.002366 )
		),
		cms.PSet(
			calo = cms.string( "hcal" ),
			iEta = cms.string( "er10to12" ),
			params = cms.vdouble( -0.004619, 0.003095 )
		),
		cms.PSet(
			calo = cms.string( "hcal" ),
			iEta = cms.string( "er13to15" ),
			params = cms.vdouble( -0.005728, 0.004538 )
		),
		cms.PSet(
			calo = cms.string( "hcal" ),
			iEta = cms.string( "er16to18" ),
			params = cms.vdouble( -0.005151, 0.001507 )
		),
		cms.PSet(
			calo = cms.string( "hgcalEM" ),
			iEta = cms.string( "er1p4to1p8" ),
			params = cms.vdouble( -0.020608, 0.004124 )
		),
		cms.PSet(
			calo = cms.string( "hgcalEM" ),
			iEta = cms.string( "er1p8to2p1" ),
			params = cms.vdouble( -0.027428, 0.005488 )
		),
		cms.PSet(
			calo = cms.string( "hgcalEM" ),
			iEta = cms.string( "er2p1to2p4" ),
			params = cms.vdouble( -0.029345, 0.005871 )
		),
		cms.PSet(
			calo = cms.string( "hgcalEM" ),
			iEta = cms.string( "er2p4to2p7" ),
			params = cms.vdouble( -0.028139, 0.005630 )
		),
		cms.PSet(
			calo = cms.string( "hgcalEM" ),
			iEta = cms.string( "er2p7to3p1" ),
			params = cms.vdouble( -0.025012, 0.005005 )
		),
		cms.PSet(
			calo = cms.string( "hgcalHad" ),
			iEta = cms.string( "er1p4to1p8" ),
			params = cms.vdouble( -0.003102, 0.000622 )
		),
		cms.PSet(
			calo = cms.string( "hgcalHad" ),
			iEta = cms.string( "er1p8to2p1" ),
			params = cms.vdouble( -0.003454, 0.000693 )
		),
		cms.PSet(
			calo = cms.string( "hgcalHad" ),
			iEta = cms.string( "er2p1to2p4" ),
			params = cms.vdouble( -0.004145, 0.000831 )
		),
		cms.PSet(
			calo = cms.string( "hgcalHad" ),
			iEta = cms.string( "er2p4to2p7" ),
			params = cms.vdouble( -0.004486, 0.000899 )
		),
		cms.PSet(
			calo = cms.string( "hgcalHad" ),
			iEta = cms.string( "er2p7to3p1" ),
			params = cms.vdouble( -0.010332, 0.002068 )
		),
		cms.PSet(
			calo = cms.string( "hf" ),
			iEta = cms.string( "er29to33" ),
			params = cms.vdouble( -0.108537, 0.021707 )
		),
		cms.PSet(
			calo = cms.string( "hf" ),
			iEta = cms.string( "er34to37" ),
			params = cms.vdouble( -0.102821, 0.020566 )
		),
		cms.PSet(
			calo = cms.string( "hf" ),
			iEta = cms.string( "er38to41" ),
			params = cms.vdouble( -0.109859, 0.021974 )
		)
	)
)
