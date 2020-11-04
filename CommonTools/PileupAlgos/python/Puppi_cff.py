import FWCore.ParameterSet.Config as cms

puppiCentral = cms.VPSet(
                 cms.PSet(
                  algoId           = cms.int32(5),  #0 is default Puppi
                  useCharged       = cms.bool(True),
                  applyLowPUCorr   = cms.bool(True),
                  combOpt          = cms.int32(0),
                  cone             = cms.double(0.4),
                  rmsPtMin         = cms.double(0.1),
                  rmsScaleFactor   = cms.double(1.0)
                 )
                )

puppiForward = cms.VPSet(
                cms.PSet(
                 algoId         = cms.int32(5),  #0 is default Puppi
                 useCharged     = cms.bool(False),
                 applyLowPUCorr = cms.bool(True),
                 combOpt        = cms.int32(0),
                 cone           = cms.double(0.4),
                 rmsPtMin       = cms.double(0.5),
                 rmsScaleFactor = cms.double(1.0)
                 )
                )

puppi = cms.EDProducer("PuppiProducer",#cms.PSet(#"PuppiProducer",
                       puppiDiagnostics = cms.bool(False),
                       puppiNoLep = cms.bool(False),
                       UseFromPVLooseTight = cms.bool(False),
                       UseDeltaZCut   = cms.bool(True),
                       EtaMinUseDeltaZ = cms.double(2.4),
                       DeltaZCut      = cms.double(0.3),
                       NumOfPUVtxsForCharged = cms.uint32(0),
                       DeltaZCutForChargedFromPUVtxs = cms.double(0.2),
		       PtMaxCharged   = cms.double(20.),
		       EtaMaxCharged   = cms.double(99999.),
		       PtMaxPhotons = cms.double(-1.),
		       EtaMaxPhotons = cms.double(2.5),
		       PtMaxNeutrals  = cms.double(200.),
		       PtMaxNeutralsStartSlope = cms.double(20.),
                       candName       = cms.InputTag('particleFlow'),
                       vertexName     = cms.InputTag('offlinePrimaryVertices'),
                       #candName      = cms.string('packedPFCandidates'),
                       #vertexName     = cms.string('offlineSlimmedPrimaryVertices'),
                       applyCHS       = cms.bool  (True),
                       invertPuppi    = cms.bool  (False),
                       useExp         = cms.bool  (False),
                       MinPuppiWeight = cms.double(0.01),
                       useExistingWeights = cms.bool(False),
                       clonePackedCands   = cms.bool(False), # should only be set to True for MiniAOD
                       vtxNdofCut     = cms.int32(4),
                       vtxZCut        = cms.double(24),
                       algos          = cms.VPSet( 
                        cms.PSet( 
                         etaMin = cms.vdouble(0.),
                         etaMax = cms.vdouble(2.5),
                         ptMin  = cms.vdouble(0.),
                         MinNeutralPt   = cms.vdouble(0.2),
                         MinNeutralPtSlope   = cms.vdouble(0.015),
                         RMSEtaSF = cms.vdouble(1.0),
                         MedEtaSF = cms.vdouble(1.0),
                         EtaMaxExtrap = cms.double(2.0),
                         puppiAlgos = puppiCentral
                        ),
                        cms.PSet( 
                         etaMin              = cms.vdouble( 2.5,  3.0),
                         etaMax              = cms.vdouble( 3.0, 10.0),
                         ptMin               = cms.vdouble( 0.0,  0.0),
                         MinNeutralPt        = cms.vdouble( 1.7,  2.0),
                         MinNeutralPtSlope   = cms.vdouble(0.08, 0.08),
                         RMSEtaSF            = cms.vdouble(1.20, 0.95),
                         MedEtaSF            = cms.vdouble(0.90, 0.75),
                         EtaMaxExtrap        = cms.double( 2.0),
                         puppiAlgos = puppiForward
                        ),
                       #  cms.PSet( 
                       #   etaMin = cms.double(3.0),
                       #   etaMax = cms.double(10.0),
                       #   ptMin  = cms.double(0.0),
                       #   MinNeutralPt        = cms.double(2.0),
                       #   MinNeutralPtSlope   = cms.double(0.07),
                       #   # RMSEtaSF = cms.double(1.18),
                       #   # MedEtaSF = cms.double(0.4397),                         
                       #   RMSEtaSF = cms.double(1.10),
                       #   MedEtaSF = cms.double(0.90),
                       #   EtaMaxExtrap = cms.double(2.0),
                       #   puppiAlgos = puppiForward
                       # )
                      )
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(
    puppi,
    EtaMinUseDeltaZ = -1.,
    PtMaxCharged = -1.,
    PtMaxNeutralsStartSlope = 0.,
    DeltaZCut = cms.double(0.1),
    algos = cms.VPSet( 
        cms.PSet( 
             etaMin = cms.vdouble(0.,  2.5),
             etaMax = cms.vdouble(2.5, 3.5),
             ptMin  = cms.vdouble(0.,  0.), #Normally 0
             MinNeutralPt   = cms.vdouble(0.2, 0.2),
             MinNeutralPtSlope   = cms.vdouble(0.015, 0.030),
             RMSEtaSF = cms.vdouble(1.0, 1.0),
             MedEtaSF = cms.vdouble(1.0, 1.0),
             EtaMaxExtrap = cms.double(2.0),
             puppiAlgos = puppiCentral
        ), cms.PSet( 
             etaMin              = cms.vdouble( 3.5),
             etaMax              = cms.vdouble(10.0),
             ptMin               = cms.vdouble( 0.), #Normally 0
             MinNeutralPt        = cms.vdouble( 2.0),
             MinNeutralPtSlope   = cms.vdouble(0.08),
             RMSEtaSF            = cms.vdouble(1.0 ),
             MedEtaSF            = cms.vdouble(0.75),
             EtaMaxExtrap        = cms.double( 2.0),
             puppiAlgos = puppiForward
       )
    )
)

puppiNoLep = puppi.clone(
    puppiNoLep = True,
    PtMaxPhotons = 20.
    )
