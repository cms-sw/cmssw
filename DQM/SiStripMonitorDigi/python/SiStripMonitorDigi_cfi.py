import FWCore.ParameterSet.Config as cms

# SiStripMonitorDigi
SiStripMonitorDigi = cms.EDFilter("SiStripMonitorDigi",
    CreateTrendMEs = cms.bool(False),
    # by default do not write out any file with histograms
    # can overwrite this in .cfg file with: replace SiStripMonitorDigi.OutputMEsInRootFile = true
    OutputMEsInRootFile = cms.bool(False),
    TH1ADCsCoolestStrip = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(False),
        Nbinx = cms.int32(60),
        xmax = cms.double(299.5),
        moduleswitchon = cms.bool(True)
    ),
    # add digi producers same way as Domenico in SiStripClusterizer
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('ZeroSuppressed'),
        DigiProducer = cms.string('siStripDigis')
    ), 
        cms.PSet(
            DigiLabel = cms.string('VirginRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ProcessedRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ScopeMode'),
            DigiProducer = cms.string('siStripZeroSuppression')
        )),
    #select detectors
    detectorson = cms.PSet(
        tidon = cms.bool(True),
        tibon = cms.bool(True),
        tecon = cms.bool(True),
        tobon = cms.bool(True)
    ),
    TH1ADCsHottestStrip = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(False),
        Nbinx = cms.int32(60),
        xmax = cms.double(299.5),
        moduleswitchon = cms.bool(True)
    ),
    OutputFileName = cms.string('/tmp/charaf/test_digi_sim.root'),
    # rest of parameters
    SelectAllDetectors = cms.bool(False),
    CalculateStripOccupancy = cms.bool(False),
    Trending = cms.PSet(
        UpdateMode = cms.int32(1),
        Nbins = cms.int32(10),
        ymax = cms.double(10000.0),
        Steps = cms.int32(10),
        xmax = cms.double(10.0),
        xmin = cms.double(0.0),
        ymin = cms.double(0.0)
    ),
    TH1DigiADCs = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(True),
        Nbinx = cms.int32(100),
        xmax = cms.double(499.5),
        moduleswitchon = cms.bool(True)
    ),
    ShowControlView = cms.bool(False),
    TH1NumberOfDigis = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(True),
        Nbinx = cms.int32(50),
        xmax = cms.double(499.5),
        moduleswitchon = cms.bool(True)
    ),
    ResetMEsEachRun = cms.bool(False),
    TH1StripOccupancy = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(100),
        xmax = cms.double(499.5)
    ),
    ShowReadoutView = cms.bool(False),
    ShowMechanicalStructureView = cms.bool(True)
)


