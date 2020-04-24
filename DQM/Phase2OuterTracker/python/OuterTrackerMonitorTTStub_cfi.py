import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorTTStub = cms.EDAnalyzer('OuterTrackerMonitorTTStub',
    
    TopFolderName = cms.string('Phase2OuterTracker'),
    TTStubs       = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),


# TTStub barrel y vs x
# TTStub forward/backward endcap y vs x
    TH2TTStub_Position = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(120),
        xmin = cms.double(-120),
        Nbinsy = cms.int32(960),
        ymax = cms.double(120),
        ymin = cms.double(-120)
        ),

#TTStub barrel y vs x zoomed
    TH2TTStub_Barrel_XY_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(60),
        xmin = cms.double(30),
        Nbinsy = cms.int32(960),
        ymax = cms.double(15),
        ymin = cms.double(-15)
        ),

#TTStub #rho vs z
    TH2TTStub_RZ = cms.PSet(
        Nbinsx = cms.int32(900),
        xmax = cms.double(300),
        xmin = cms.double(-300),
        Nbinsy = cms.int32(900),
        ymax = cms.double(120),
        ymin = cms.double(0)
        ),

#TTStub Forward Endcap #rho vs. z
    TH2TTStub_Endcap_Fw_RZ_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(170),
        xmin = cms.double(140),
        Nbinsy = cms.int32(960),
        ymax = cms.double(60),
        ymin = cms.double(30)
        ),

#TTStub Backward Endcap #rho vs. z
    TH2TTStub_Endcap_Bw_RZ_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(-140),
        xmin = cms.double(-170),
        Nbinsy = cms.int32(960),
        ymax = cms.double(100),
        ymin = cms.double(70)
        ),

#TTStub eta distribution
    TH1TTStub_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmin = cms.double(-3),
        xmax = cms.double(3)
        ),

#TTStub Barrel Layers
    TH1TTStub_Layers = cms.PSet(
        Nbinsx = cms.int32(6),
        xmin = cms.double(0.5),
        xmax = cms.double(6.5)
        ),

#TTStub EC Discs
    TH1TTStub_Discs = cms.PSet(
        Nbinsx = cms.int32(5),
        xmin = cms.double(0.5),
        xmax = cms.double(5.5)
        ),

#TTStub EC Rings
    TH1TTStub_Rings = cms.PSet(
        Nbinsx = cms.int32(16),
        xmin = cms.double(0.5),
        xmax = cms.double(16.5)
        ),

#TTStub displacement or offset per Layer
    TH2TTStub_DisOf_Layer = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),
        xmin = cms.double(0.5),
        Nbinsy = cms.int32(43),
        ymax = cms.double(10.75),
        ymin = cms.double(-10.75)
        ),

#TTStub displacement or offset per Disc
    TH2TTStub_DisOf_Disc = cms.PSet(
        Nbinsx = cms.int32(5),
        xmax = cms.double(5.5),
        xmin = cms.double(0.5),
        Nbinsy = cms.int32(43),
        ymax = cms.double(10.75),
        ymin = cms.double(-10.75)
        ),

#TTStub displacement or offset per Ring
    TH2TTStub_DisOf_Ring = cms.PSet(
        Nbinsx = cms.int32(16),
        xmax = cms.double(16.5),
        xmin = cms.double(0.5),
        Nbinsy = cms.int32(43),
        ymax = cms.double(10.75),
        ymin = cms.double(-10.75)
        ),

)
