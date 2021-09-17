import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
OuterTrackerMonitorTTStub = DQMEDAnalyzer('OuterTrackerMonitorTTStub',

    TopFolderName = cms.string('SiOuterTracker'),
    TTStubs       = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),

# TTStub forward/backward endcap y vs x
    TH2TTStub_Position = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(120),
        xmin = cms.double(-120),
        Nbinsy = cms.int32(960),
        ymax = cms.double(120),
        ymin = cms.double(-120)
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

#TTStub eta distribution
    TH1TTStub_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmin = cms.double(-5),
        xmax = cms.double(5)
        ),

#TTStub phi distribution
    TH1TTStub_Phi = cms.PSet(
        Nbinsx = cms.int32(60),
        xmin = cms.double(-3.5),
        xmax = cms.double(3.5)
        ),

#TTStub R distribution
    TH1TTStub_R = cms.PSet(
        Nbinsx = cms.int32(45),
        xmin = cms.double(0),
        xmax = cms.double(120)
        ),

#TTStub bend distribution
    TH1TTStub_bend = cms.PSet(
        Nbinsx = cms.int32(69),
        xmin = cms.double(-8.625),
        xmax = cms.double(8.625)
        ),

#TTStub, isPS?
    TH1TTStub_isPS = cms.PSet(
        Nbinsx = cms.int32(2),
        xmin = cms.double(0.0),
        xmax = cms.double(2.0)
        ),

#TTStub Barrel Layers
    TH1TTStub_Layers = cms.PSet(
        Nbinsx = cms.int32(7),
        xmin = cms.double(0.5),
        xmax = cms.double(7.5)
        ),

#TTStub EC Discs
    TH1TTStub_Discs = cms.PSet(
        Nbinsx = cms.int32(6),
        xmin = cms.double(0.5),
        xmax = cms.double(6.5)
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
