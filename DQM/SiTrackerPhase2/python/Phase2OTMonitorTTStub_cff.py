import FWCore.ParameterSet.Config as cms

from DQM.SiTrackerPhase2.Phase2OTMonitorTTStub_cfi import Phase2OTMonitorTTStub as Phase2OTMonitorTTStub
TTStubMonitor = Phase2OTMonitorTTStub.clone(
    CrackOverview = Phase2OTMonitorTTStub.CrackOverview.clone(
        switch = cms.bool(False)
    )
)
TTStubMonitorCRACK = Phase2OTMonitorTTStub.clone(
    # Histograms that are usually set to switch = False in full tracker
    CrackOverview = Phase2OTMonitorTTStub.CrackOverview.clone(
        name = cms.string('Crack_Overview_stubs'),
        title = cms.string('Crack_Overview_stubs;Module;Layer'),
        xmin = cms.double(0.0),
        xmax = cms.double(13.0),
        ymin = cms.double(0.0),
        ymax = cms.double(7.5),
        switch = cms.bool(True)
    ),
    # Changes to x/y ranges for CRACK readability
    L1Stub_Global_Position_Barrel_XY = Phase2OTMonitorTTStub.L1Stub_Global_Position_Barrel_XY.clone(
        xmin = cms.double(-7.0),
        xmax = cms.double(7.0),
        ymin = cms.double(-10.0),
        ymax = cms.double(50.0)
    ),
    L1Stub_Global_Position_RZ = Phase2OTMonitorTTStub.L1Stub_Global_Position_RZ.clone(
        xmin = cms.double(-70.0),
        xmax = cms.double(70.0),
        ymin = cms.double(-0.0),
        ymax = cms.double(60.0)
    )
)
