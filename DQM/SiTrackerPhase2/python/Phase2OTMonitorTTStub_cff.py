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
        xmin = cms.double(-10.0),
        xmax = cms.double(130.0),
        NxBins = cms.int32(70),
        ymin = cms.double(-5.0),
        ymax = cms.double(50.0),
        NyBins = cms.int32(55)
    ),
    NStubsLayer = Phase2OTMonitorTTStub.NStubsLayer.clone(
        xmin = cms.double(-0.5),
        xmax = cms.double(50.5),
        NxBins = cms.int32(51)
    ),
    # Removing endcap histograms
    L1Stub_Global_Position_Endcap_Bw_XY = Phase2OTMonitorTTStub.L1Stub_Global_Position_Endcap_Bw_XY.clone(
        switch = cms.bool(False)
    ),
    L1Stub_Global_Position_Endcap_Fw_XY = Phase2OTMonitorTTStub.L1Stub_Global_Position_Endcap_Fw_XY.clone(
        switch = cms.bool(False)
    )
)
