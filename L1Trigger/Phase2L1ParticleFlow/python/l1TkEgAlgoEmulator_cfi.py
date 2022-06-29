import FWCore.ParameterSet.Config as cms


tkEgAlgoParameters = cms.PSet(
    nTRACK=cms.uint32(50),  # very large numbers for first test
    nTRACK_EGIN=cms.uint32(50),  # very large numbers for first test
    nEMCALO_EGIN=cms.uint32(50),  # very large numbers for first test
    nEM_EGOUT=cms.uint32(50),  # very large numbers for first test
    doBremRecovery=cms.bool(False),
    writeBeforeBremRecovery=cms.bool(True),
    filterHwQuality=cms.bool(False),
    caloHwQual=cms.int32(4),
    doEndcapHwQual=cms.bool(False),
    dEtaMaxBrem=cms.double(0.02),
    dPhiMaxBrem=cms.double(0.1),
    absEtaBoundaries=cms.vdouble(0.0, 0.9, 1.5),
    dEtaValues=cms.vdouble(0.025, 0.015, 0.01),  # last was  0.0075  in TDR
    dPhiValues=cms.vdouble(0.07, 0.07, 0.07),
    caloEtMin=cms.double(0.0),
    trkQualityPtMin=cms.double(10.0),
    writeEGSta=cms.bool(False),
    tkIsoParametersTkEm=cms.PSet(
        tkQualityPtMin=cms.double(2.),
        dZ=cms.double(0.6),
        dRMin=cms.double(0.07),
        dRMax=cms.double(0.30),
        tkQualityChi2Max=cms.double(100),
    ),
    tkIsoParametersTkEle=cms.PSet(
        tkQualityPtMin=cms.double(2.),
        dZ=cms.double(0.6),
        dRMin=cms.double(0.03),
        dRMax=cms.double(0.20),
        tkQualityChi2Max=cms.double(1e10),
    ),
    pfIsoParametersTkEm=cms.PSet(
        tkQualityPtMin=cms.double(1.),
        dZ=cms.double(0.6),
        dRMin=cms.double(0.07),
        dRMax=cms.double(0.30),
        tkQualityChi2Max=cms.double(100),
    ),
    pfIsoParametersTkEle=cms.PSet(
        tkQualityPtMin=cms.double(1.),
        dZ=cms.double(0.6),
        dRMin=cms.double(0.03),
        dRMax=cms.double(0.20),
        tkQualityChi2Max=cms.double(1e10),
    ),
    doTkIso=cms.bool(True),
    doPfIso=cms.bool(True),
    hwIsoTypeTkEle=cms.uint32(0),
    hwIsoTypeTkEm=cms.uint32(2),
    doCompositeTkEle=cms.bool(False),
    compositeParametersTkEle=cms.PSet( # Parameters used to normalize input features
        hoeMin=cms.double(-1.0),
        hoeMax=cms.double(1566.547607421875),
        tkptMin=cms.double(1.9501149654388428),
        tkptMax=cms.double(11102.0048828125),
        srrtotMin=cms.double(0.0),
        srrtotMax=cms.double(0.01274710614234209),
        detaMin=cms.double(-0.24224889278411865),
        detaMax=cms.double(0.23079538345336914),
        dptMin=cms.double(0.010325592942535877),
        dptMax=cms.double(184.92538452148438),
        meanzMin=cms.double(325.0653991699219),
        meanzMax=cms.double(499.6089782714844),
        dphiMin=cms.double(-6.281332015991211),
        dphiMax=cms.double(6.280326843261719),
        tkchi2Min=cms.double(0.024048099294304848),
        tkchi2Max=cms.double(1258.37158203125),
        tkz0Min=cms.double(-14.94140625),
        tkz0Max=cms.double(14.94140625),
        tknstubsMin=cms.double(4.0),
        tknstubsMax=cms.double(6.0),
        BDTcut_wp97p5=cms.double(0.5406244),
        BDTcut_wp95p0=cms.double(0.9693441),
    ),
)

tkEgSorterParameters = cms.PSet(
    nObjToSort=cms.uint32(6),
    nObjSorted=cms.uint32(16),
)
