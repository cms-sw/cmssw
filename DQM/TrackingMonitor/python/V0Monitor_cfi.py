import FWCore.ParameterSet.Config as cms

v0Monitor = cms.EDAnalyzer("V0Monitor",
   FolderName    = cms.string("Tracking/V0Monitoring"),
   v0            = cms.InputTag('generalV0Candidates:Kshort'), # generalV0Candidates:Lambda
   beamSpot      = cms.InputTag('offlineBeamSpot'),
   primaryVertex = cms.InputTag('offlinePrimaryVertices'),
   lumiScalers   = cms.InputTag('scalersRawToDigi'),
   pvNDOF = cms.int32(4),   
   genericTriggerEventPSet = cms.PSet(),
   histoPSet = cms.PSet(
      lumiPSet = cms.PSet(
            nbins = cms.int32 ( 3700 ),
            xmin  = cms.double(    0.),
            xmax  = cms.double(14000.),
      ),
      massPSet = cms.PSet(
            nbins = cms.int32 ( 100 ),
            xmin  = cms.double( 0.400), # 1.050
            xmax  = cms.double( 0.600), # 1.250
      ),
      ptPSet = cms.PSet(
            nbins = cms.int32 ( 100 ),
            xmin  = cms.double(   0.),
            xmax  = cms.double(  50.),
      ),
      etaPSet = cms.PSet(
            nbins = cms.int32 ( 60 ),
            xmin  = cms.double( -3.),
            xmax  = cms.double(  3.),
      ),
      LxyPSet = cms.PSet(
            nbins = cms.int32 (350),
            xmin  = cms.double(  0.),
            xmax  = cms.double( 70.),
      ),
      chi2oNDFPSet = cms.PSet(
            nbins = cms.int32 (100 ),
            xmin  = cms.double(  0.),
            xmax  = cms.double( 30.),
      ),
      puPSet = cms.PSet(
            nbins = cms.int32 ( 60 ),
            xmin  = cms.double( -0.5),
            xmax  = cms.double( 59.5),
      ),
      lsPSet = cms.PSet(
            nbins = cms.int32 ( 2000 ),
            xmin  = cms.double(    0.),
            xmax  = cms.double( 2000.),
      ),
   ),
)
