import FWCore.ParameterSet.Config as cms

puppiCentral = cms.VPSet(
                 cms.PSet(
                  algoId           = cms.int32(5),  #0 is default Puppi
                  useCharged       = cms.bool(True),
                  applyLowPUCorr   = cms.bool(True),
                  combOpt          = cms.int32(0),
                  cone             = cms.double(0.3),
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
                 cone           = cms.double(0.3),
                 rmsPtMin       = cms.double(0.1),
                 rmsScaleFactor = cms.double(1.0)
                 )
                )

puppi = cms.EDProducer("PuppiProducer",#cms.PSet(#"PuppiProducer",
                       puppiDiagnostics = cms.bool(False),
                       UseDeltaZCut   = cms.bool(False),
                       DeltaZCut      = cms.double(0.3),
                       candName       = cms.InputTag('particleFlow'),
                       vertexName     = cms.InputTag('offlinePrimaryVertices'),
                       #candName      = cms.string('packedPFCandidates'),
                       #vertexName     = cms.string('offlineSlimmedPrimaryVertices'),
                       applyCHS       = cms.bool  (True),
                       useExp         = cms.bool  (False),
                       MinPuppiWeight = cms.double(0.01),
                       algos          = cms.VPSet( 
                        cms.PSet( 
                         etaMin = cms.double(0.),
                         etaMax = cms.double( 2.0),
                         ptMin  = cms.double(0.),
                         MinNeutralPt   = cms.double(0.1),
                         MinNeutralPtSlope   = cms.double(0.005),
                         RMSEtaSF = cms.double(1.0),
                         MedEtaSF = cms.double(1.0),
                         EtaMaxExtrap = cms.double(2.0),
                         puppiAlgos = puppiCentral
                        ),
                        cms.PSet( 
                         etaMin = cms.double(2.0),
                         etaMax = cms.double( 2.5),
                         ptMin  = cms.double(0.),
                         MinNeutralPt   = cms.double(0.1),
                         MinNeutralPtSlope   = cms.double(0.005),
                         RMSEtaSF = cms.double(0.95),
                         MedEtaSF = cms.double(1.0),
                         EtaMaxExtrap = cms.double(2.0),
                         puppiAlgos = puppiCentral
                        ),
                        cms.PSet( 
                         etaMin = cms.double(2.5),
                         etaMax = cms.double(3.0),
                         ptMin  = cms.double(0.0),
                         MinNeutralPt        = cms.double(1.0),
                         MinNeutralPtSlope   = cms.double(0.005),
                         # RMSEtaSF = cms.double(1.545),
                         # MedEtaSF = cms.double(0.845),
                         RMSEtaSF = cms.double(1.484),
                         MedEtaSF = cms.double(0.845),
                         EtaMaxExtrap = cms.double(2.0),
                         puppiAlgos = puppiForward
                        ),
                        cms.PSet( 
                         etaMin = cms.double(3.0),
                         etaMax = cms.double(10.0),
                         ptMin  = cms.double(0.0),
                         MinNeutralPt        = cms.double(1.0),
                         MinNeutralPtSlope   = cms.double(0.005),
                         # RMSEtaSF = cms.double(1.18),
                         # MedEtaSF = cms.double(0.4397),                         
                         RMSEtaSF = cms.double(1.25),
                         MedEtaSF = cms.double(0.4397),
                         EtaMaxExtrap = cms.double(2.0),
                         puppiAlgos = puppiForward
                       )
                      )
)
