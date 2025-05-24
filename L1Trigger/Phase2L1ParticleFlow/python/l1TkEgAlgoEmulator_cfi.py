import FWCore.ParameterSet.Config as cms


CompositeParametersTkEleVec = cms.VPSet(
    #Algorithm 0:  Elliptic ID
    cms.PSet( # this is not used by any CompositeID model!
        model=cms.string(""),
        loose_wp=cms.PSet(
            bins=cms.vdouble(0),
            values=cms.vdouble(-1)
            ),
        tight_wp=cms.PSet(
            bins=cms.vdouble(0),
            values=cms.vdouble(-1)
            ),
        dEta_max = cms.double(-1),
        dPhi_max = cms.double(-1),
    ),
    #Algorithm 1: compositeEE_v0
    cms.PSet(
        # NOTE: conifer BDT score is log(p/1-p)
        # the working points are cuts on BDT output logits [log(p/1-p)]/4 (range -1 to 1 to match the FW dataformat)
        model=cms.string("L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EE_v0.json"),
        loose_wp=cms.PSet(
            bins=cms.vdouble(0., 18., 28., 36.),
            values=cms.vdouble(-0.181641, -0.15, 0.075, -0.181641)
            ),
        tight_wp=cms.PSet(
            bins=cms.vdouble(0., 28., 40.),
            values=cms.vdouble(0.0527344, 0.3, 0.0527344)
            ),
        dPhi_max = cms.double(0.2),
        dEta_max = cms.double(0.2),
    ),
    #Algorithm 2: compositeEB_v0
    cms.PSet(
        model=cms.string("L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EB_v0.json"),
        loose_wp=cms.PSet(
            bins=cms.vdouble(0),
            values=cms.vdouble(-1)
            ),
        tight_wp=cms.PSet(
            bins=cms.vdouble(0, 5, 10, 20, 30, 50),
            values=cms.vdouble(0.19, 0.05, -0.35, -0.45, -0.5, -0.38),
        ),
        dPhi_max = cms.double(0.3),
        dEta_max = cms.double(0.03),
    ),
    #Algorithm 3: compositeEE_v1
    cms.PSet(
        model=cms.string("L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EE_v1.json"),
        loose_wp=cms.PSet(
            bins=cms.vdouble(0),
            values=cms.vdouble(-1)
            ),
        tight_wp=cms.PSet(
            bins=cms.vdouble(0, 5, 10, 20, 30, 50),
            values=cms.vdouble(0.309191, 0.0909913, -0.211824, -0.223846, -0.188443, -0.154760),
        ),
        dPhi_max = cms.double(0.3),
        dEta_max = cms.double(0.03),
    ),
    #Algorithm 4: compositeEB_v1
    cms.PSet(
        model=cms.string("L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EB_v1.json"),
        loose_wp=cms.PSet(
            bins=cms.vdouble(0),
            values=cms.vdouble(-1)
            ),
        tight_wp=cms.PSet(
            bins=cms.vdouble(0, 5, 10, 20, 30, 50),
            values=cms.vdouble(0.17, 0.018, -0.08, -0.11, -0.215, -0.15),
        ),
        dPhi_max = cms.double(0.3),
        dEta_max = cms.double(0.03),
    )
)



tkEgAlgoParameters = cms.PSet(
    # debug=cms.untracked.uint32(4),
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
    ),
    tkIsoParametersTkEle=cms.PSet(
        tkQualityPtMin=cms.double(2.),
        dZ=cms.double(0.6),
        dRMin=cms.double(0.03),
        dRMax=cms.double(0.20),
    ),
    pfIsoParametersTkEm=cms.PSet(
        tkQualityPtMin=cms.double(1.),
        dZ=cms.double(0.6),
        dRMin=cms.double(0.07),
        dRMax=cms.double(0.30),
    ),
    pfIsoParametersTkEle=cms.PSet(
        tkQualityPtMin=cms.double(1.),
        dZ=cms.double(0.6),
        dRMin=cms.double(0.03),
        dRMax=cms.double(0.20),
    ),
    doTkIso=cms.bool(True),
    doPfIso=cms.bool(True),
    hwIsoTypeTkEle=cms.uint32(0),
    hwIsoTypeTkEm=cms.uint32(0),
    algorithm=cms.uint32(0), # 0 = elliptic , 1 = composite EE, 2 = composite EB
    nCompCandPerCluster=cms.uint32(3),
    compositeParametersTkEle=CompositeParametersTkEleVec,
)

tkEgSorterParameters = cms.PSet(
    nObjToSort=cms.uint32(6),
    nObjSorted=cms.uint32(16),
)
