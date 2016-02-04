import FWCore.ParameterSet.Config as cms

process = cms.Process("MONITOR")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
poolDBESSource.connect = "frontier://FrontierDev/CMS_COND_ALIGNMENT"
poolDBESSource.toGet = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    )) 
process.glbPositionSource = poolDBESSource

process.source = cms.Source("PoolSource",
    useCSA08Kludge = cms.untracked.bool(True),
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/0069285D-EC54-DD11-A9C7-001D09F23A84.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/081AAA0D-ED54-DD11-9C3B-000423D94E1C.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/22F99FB0-EC54-DD11-829B-001D09F24489.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/34A13084-EC54-DD11-8D47-001D09F241D2.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/3AC99F3C-EC54-DD11-9428-0019B9F72F97.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/3AE9EF62-EC54-DD11-AB8F-001D09F2A465.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/3EA06487-4255-DD11-8231-000423D99996.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/5037EC61-EC54-DD11-8125-0030487A18A4.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/523F4987-EC54-DD11-8171-001D09F23C73.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/56913919-EC54-DD11-A76B-001D09F23A84.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/74AE0C8D-EC54-DD11-9F88-0019B9F704D1.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/78D09F3D-EC54-DD11-9BF3-001D09F241B4.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/8E56A436-EE54-DD11-AAD0-000423D6AF24.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/A64CB69F-EC54-DD11-9900-001D09F24FBA.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/B4C6DDB3-EC54-DD11-860C-001D09F251CC.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/B879C75E-EC54-DD11-A1FB-001D09F254CE.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/C01ED3FF-EC54-DD11-994A-001D09F29146.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/D89C578B-EC54-DD11-A58F-001D09F2532F.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/E6DF7A3A-EC54-DD11-BB6C-001D09F25217.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/F4533760-EC54-DD11-9C06-001D09F291D2.root',
        '/store/data/CRUZET3/TestEnables/RAW/v4/000/051/384/F6659FE8-EC54-DD11-A70C-001617C3B6CE.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    useStandardFEDid = cms.untracked.bool(True),
    fedbyType = cms.untracked.bool(True),
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    )
)

process.DTMapping = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTReadOutMappingRcd'),
        tag = cms.string('map_CRUZET')
    ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('tTrig_CRUZET_080708_2019')
        )),
    connect = cms.string('frontier://FrontierProd/CMS_COND_20X_DT')
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    authenticationMethod = cms.untracked.uint32(0),
    connect = cms.string('sqlite_file:t0new.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0new')
    ))
)

process.t0calib = cms.EDAnalyzer("DTT0CalibrationNew",
    # Cells for which you want the histos (default = None)
    cellsWithHisto = cms.untracked.vstring('-1 8 1 1 3 48', 
        '-1 8 1 1 3 49', 
        '-1 8 1 1 2 49', 
        '-1 8 1 1 2 50', 
        '-1 8 1 1 1 48', 
        '-1 8 1 1 1 49'),
    # Criteria to reject digis away from TP peak
    rejectDigiFromPeak = cms.uint32(50),
    # Label to retrieve DT digis from the event
    digiLabel = cms.untracked.string('dtunpacker'),
    calibSector = cms.untracked.string('All'),
    # Chose the wheel, sector (default = All)
    calibWheel = cms.untracked.string('All'),
    # Number of events to be used for the t0 per layer histos
    eventsForWireT0 = cms.uint32(600),
    # Name of the ROOT file which will contain the test pulse times per layer
    rootFileName = cms.untracked.string('DTTestPulses.root'),
    debug = cms.untracked.bool(True),
    # Acceptance for TP peak width
    tpPeakWidth = cms.double(5.0),
    # Acceptance for TP peak width per Layer
    tpPeakWidthPerLayer = cms.double(10.0),
    # Time box width (TP within time box)
    timeBoxWidth = cms.uint32(500),
    # Number of events to be used for the t0 per layer histos
    eventsForLayerT0 = cms.uint32(400)
)

process.p = cms.Path(process.dtunpacker*process.t0calib)


