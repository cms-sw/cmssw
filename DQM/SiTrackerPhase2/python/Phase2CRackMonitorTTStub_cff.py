import FWCore.ParameterSet.Config as cms

from DQM.SiTrackerPhase2.Phase2OTMonitorTTStub_cfi import Phase2OTMonitorTTStub as _Phase2OTMonitorTTStub

TTStubMonitorCRACK = _Phase2OTMonitorTTStub.clone(
    # Histograms that are usually set to switch = False in full tracker
    CrackOverview = _Phase2OTMonitorTTStub.CrackOverview.clone(
        name = cms.string('Crack_Overview_stubs'),
        title = cms.string('Crack_Overview_stubs;Module;Layer'),
        xmin = cms.double(0.0),
        xmax = cms.double(13.0),
        ymin = cms.double(0.0),
        ymax = cms.double(7.5),
        switch = cms.bool(True)
    ),
    # Changes to x/y ranges for CRACK readability
    L1Stub_Global_Position_Barrel_XY = _Phase2OTMonitorTTStub.L1Stub_Global_Position_Barrel_XY.clone(
        xmin = cms.double(-7.0),
        xmax = cms.double(7.0),
        ymin = cms.double(-10.0),
        ymax = cms.double(50.0)
    ),
    L1Stub_Global_Position_RZ = _Phase2OTMonitorTTStub.L1Stub_Global_Position_RZ.clone(
        xmin = cms.double(-70.0),
        xmax = cms.double(70.0),
        ymin = cms.double(-0.0),
        ymax = cms.double(60.0)
    )

    #TopFolderName = cms.string('TrackerPhase2OTCluster'),
    #clusterSrc = cms.InputTag('siPhase2Clusters'),
    #mightGet = cms.optional.untracked.vstring
)
