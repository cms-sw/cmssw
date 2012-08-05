import FWCore.ParameterSet.Config as cms

dEdxAnalyzer = cms.EDAnalyzer("dEdxAnalyzer",
    dEdxParameters = cms.PSet(
       doAllPlots          = cms.bool(False),
       doDeDxPlots         = cms.bool(True),
       FolderName          = cms.string('Tracking/dEdx'),
       OutputMEsInRootFile = cms.bool(False),
       OutputFileName      = cms.string('MonitorTrack.root'),
       
       #input collections
       TracksForDeDx       = cms.string('RefitterForDedxDQMDeDx'),
       deDxProducers       = cms.vstring('dedxDQMHarm2SP', 'dedxDQMHarm2SO', 'dedxDQMHarm2PO'),

       #cuts on number of hits
       TrackHitMin         = cms.double(8),
       HIPdEdxMin          = cms.double(3.5),

       #constants for dEdx mass reco
       dEdxK               = cms.double(2.529),
       dEdxC               = cms.double(2.772),

       #histograms definition
       dEdxNHitBin         = cms.int32(30),
       dEdxNHitMin         = cms.double(0),
       dEdxNHitMax         = cms.double(30.),

       dEdxBin             = cms.int32(100),
       dEdxMin             = cms.double(0),
       dEdxMax             = cms.double(10.),

       # MIP
       dEdxMIPmassBin      = cms.int32(100),
       dEdxMIPmassMin      = cms.double(-0.5),
       dEdxMIPmassMax      = cms.double(24.5),

       # HIP
       dEdxHIPmassBin      = cms.int32(51),
       dEdxHIPmassMin      = cms.double(-0.5),
       dEdxHIPmassMax      = cms.double(5.05),
    )                          
)
