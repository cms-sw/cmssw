import FWCore.ParameterSet.Config as cms

l1tGTAlgoBlockProducer = cms.EDProducer("L1GTAlgoBlockProducer",
    algorithms = cms.VPSet(
        cms.PSet(
            expression = cms.string('pSingleTkMuon22')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkMuon15_7')
        ),
        cms.PSet(
            expression = cms.string('pTripleTkMuon5_3_3')
        ),
        cms.PSet(
            expression = cms.string('pSingleEGEle51')
        ),
        cms.PSet(
            expression = cms.string('pDoubleEGEle37_24')
        ),
        cms.PSet(
            expression = cms.string('pIsoTkEleEGEle22_12')
        ),
        cms.PSet(
            expression = cms.string('pSingleTkEle36')
        ),
        cms.PSet(
            expression = cms.string('pSingleIsoTkEle28')
        ),
        cms.PSet(
            expression = cms.string('pSingleIsoTkPho36')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkEle25_12')
        ),
        cms.PSet(
            expression = cms.string('pDoubleIsoTkPho22_12')
        ),
        cms.PSet(
            expression = cms.string('pDoublePuppiTau52_52')
        ),
        cms.PSet(
            expression = cms.string('pSinglePuppiJet230')
        ),
        cms.PSet(
            expression = cms.string('pDoublePuppiJet112_112')
        ),
        cms.PSet(
            expression = cms.string('pDoublePuppiJet160_35_mass620')
        ),
        cms.PSet(
            expression = cms.string('pPuppiHT450')
        ),
        cms.PSet(
            expression = cms.string('pPuppiMHT140')
        ),
        cms.PSet(
            expression = cms.string('pPuppiMET200')
        ),
        cms.PSet(
            expression = cms.string('pPuppiHT400 and pQuadJet70_55_40_40'),
            name = cms.string('pPuppiHT400_pQuadJet70_55_40_40')
        ),
        cms.PSet(
            expression = cms.string('pTkMuonTkIsoEle7_20')
        ),
        cms.PSet(
            expression = cms.string('pTkMuonTkEle7_23')
        ),
        cms.PSet(
            expression = cms.string('pTkEleTkMuon10_20')
        ),
        cms.PSet(
            expression = cms.string('pTkMuonDoubleTkEle6_17_17')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkMuonTkEle5_5_9')
        ),
        cms.PSet(
            expression = cms.string('pPuppiTauTkMuon42_18')
        ),
        cms.PSet(
            expression = cms.string('pPuppiTauTkIsoEle45_22')
        ),
        cms.PSet(
            expression = cms.string('pTkMuonPuppiHT6_320')
        ),
        cms.PSet(
            expression = cms.string('pTkMuTriPuppiJet_12_40_dRMax_DoubleJet_dEtaMax')
        ),
        cms.PSet(
            expression = cms.string('pTkMuPuppiJetPuppiMet_3_110_120')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkMuPuppiJetPuppiMet_3_3_60_130')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkMuPuppiHT_3_3_300')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkElePuppiHT_8_8_390')
        ),
        cms.PSet(
            expression = cms.string('pTkEleIsoPuppiHT_26_190')
        ),
        cms.PSet(
            expression = cms.string('pTkElePuppiJet_28_40_MinDR')
        ),
        cms.PSet(
            expression = cms.string('pNNPuppiTauPuppiMet_55_190')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkMuon_OS_Er1p5_Dr1p4')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkMuon_4_4_OS_Dr1p2')
        ),
        cms.PSet(
            expression = cms.string('pDoubleTkMuon_4p5_4p5_OS_Er2_Mass7to18')
        ),
        cms.PSet(
            expression = cms.string('pTripleTkMuon_5_3_0_DoubleTkMuon_5_3_OS_MassTo9')
        ),
        cms.PSet(
            expression = cms.string('pTripleTkMuon_5_3p5_2p5_OS_Mass5to17')
        )
    )
)
