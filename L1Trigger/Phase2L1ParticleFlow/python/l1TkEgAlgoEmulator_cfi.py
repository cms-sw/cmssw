import FWCore.ParameterSet.Config as cms


tkEgAlgoParameters = cms.PSet(
    nTRACK=cms.uint32(50),  # very large numbers for first test
    nEMCALO=cms.uint32(50),  # very large numbers for first test
    nEMCALOSEL_EGIN=cms.uint32(50),  # very large numbers for first test
    nEM_EGOUT=cms.uint32(50),  # very large numbers for first test
    doBremRecovery=cms.bool(False),
    writeBeforeBremRecovery=cms.bool(True),
    filterHwQuality=cms.bool(False),
    caloHwQual=cms.int32(4),
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
    hwIsoTypeTkEm=cms.uint32(1)
)
