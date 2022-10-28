import FWCore.ParameterSet.Config as cms

hltPFPuppi = cms.EDProducer("PuppiProducer",
    DeltaZCut = cms.double(0.1),
    DeltaZCutForChargedFromPUVtxs = cms.double(0.2),
    EtaMaxCharged = cms.double(99999.0),
    EtaMaxPhotons = cms.double(2.5),
    EtaMinUseDeltaZ = cms.double(-1.0),
    MinPuppiWeight = cms.double(0.01),
    NumOfPUVtxsForCharged = cms.uint32(0),
    PUProxyValue = cms.InputTag("hltPixelClustersMultiplicity"),
    PtMaxCharged = cms.double(-1.0),
    PtMaxNeutrals = cms.double(200.0),
    PtMaxNeutralsStartSlope = cms.double(0.0),
    PtMaxPhotons = cms.double(-1.0),
    UseDeltaZCut = cms.bool(True),
    UseFromPVLooseTight = cms.bool(False),
    algos = cms.VPSet(
        cms.PSet(
            EtaMaxExtrap = cms.double(2.0),
            MedEtaSF = cms.vdouble(1.0, 1.0),
            MinNeutralPt = cms.vdouble(0.5105, 0.821),
            MinNeutralPtSlope = cms.vdouble(9.51e-06, 1.902e-05),
            RMSEtaSF = cms.vdouble(1.0, 1.0),
            etaMax = cms.vdouble(2.5, 3.5),
            etaMin = cms.vdouble(0.0, 2.5),
            ptMin = cms.vdouble(0.0, 0.0),
            puppiAlgos = cms.VPSet(cms.PSet(
                algoId = cms.int32(5),
                applyLowPUCorr = cms.bool(True),
                combOpt = cms.int32(0),
                cone = cms.double(0.4),
                rmsPtMin = cms.double(0.1),
                rmsScaleFactor = cms.double(1.0),
                useCharged = cms.bool(True)
            ))
        ),
        cms.PSet(
            EtaMaxExtrap = cms.double(2.0),
            MedEtaSF = cms.vdouble(0.75),
            MinNeutralPt = cms.vdouble(3.656),
            MinNeutralPtSlope = cms.vdouble(5.072e-05),
            RMSEtaSF = cms.vdouble(1.0),
            etaMax = cms.vdouble(10.0),
            etaMin = cms.vdouble(3.5),
            ptMin = cms.vdouble(0.0),
            puppiAlgos = cms.VPSet(cms.PSet(
                algoId = cms.int32(5),
                applyLowPUCorr = cms.bool(True),
                combOpt = cms.int32(0),
                cone = cms.double(0.4),
                rmsPtMin = cms.double(0.5),
                rmsScaleFactor = cms.double(1.0),
                useCharged = cms.bool(False)
            ))
        )
    ),
    applyCHS = cms.bool(True),
    candName = cms.InputTag("particleFlowTmp"),
    clonePackedCands = cms.bool(False),
    invertPuppi = cms.bool(False),
    puppiDiagnostics = cms.bool(False),
    puppiNoLep = cms.bool(False),
    useExistingWeights = cms.bool(False),
    useExp = cms.bool(False),
    usePUProxyValue = cms.bool(True),
    vertexName = cms.InputTag("goodOfflinePrimaryVertices"),
    vtxNdofCut = cms.int32(4),
    vtxZCut = cms.double(24)
)
