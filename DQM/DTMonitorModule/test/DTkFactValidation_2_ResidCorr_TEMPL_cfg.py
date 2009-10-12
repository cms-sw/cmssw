import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("EmptyIOVSource",
     lastValue = cms.uint64(100),
     timetype = cms.string('runnumber'),
     firstValue = cms.uint64(1),
     interval = cms.uint64(90)
 )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('resolutionTest_step1', 
        'resolutionTest_step2', 
        'resolutionTest_step3'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        #FwkJob = cms.untracked.PSet(
        #    limit = cms.untracked.int32(0)
        #),
        resolution = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('resolution'),
    destinations = cms.untracked.vstring('cout')
)

process.qTester = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests_ttrig.xml')
)
process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.modulo1=process.resolutionTest.clone()
process.modulo1.histoTag2D = 'hResDistVsDist_STEP1' 
process.modulo1.histoTag  = 'hResDist_STEP1'
process.modulo1.STEP = 'STEP1'
process.modulo1.OutputMEsInRootFile = cms.bool(False)
process.modulo1.readFile = cms.untracked.bool(True)
process.modulo1.inputFile = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/DTkFactValidation_ResidCorr_RUNNUMBERTEMPLATE.root')
 
process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.modulo2=process.resolutionTest.clone()
process.modulo2.histoTag2D = 'hResDistVsDist_STEP2' 
process.modulo2.histoTag  = 'hResDist_STEP2'
process.modulo2.STEP = 'STEP2'
process.modulo2.OutputMEsInRootFile = cms.bool(False)
process.modulo2.readFile = cms.untracked.bool(True)
process.modulo2.inputFile = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/DTkFactValidation_ResidCorr_RUNNUMBERTEMPLATE.root')
 
process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.modulo3=process.resolutionTest.clone()
process.modulo3.histoTag2D = 'hResDistVsDist_STEP3' 
process.modulo3.histoTag  = 'hResDist_STEP3'
process.modulo3.STEP = 'STEP3'
process.modulo3.OutputMEsInRootFile = cms.bool(True)
process.modulo3.readFile = cms.untracked.bool(True)
process.modulo3.inputFile = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/DTkFactValidation_ResidCorr_RUNNUMBERTEMPLATE.root')
process.modulo3.OutputFileName = cms.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/SummaryResiduals_ResidCorr_RUNNUMBERTEMPLATE.root')

process.secondStep = cms.Sequence(process.modulo1*process.modulo2*process.modulo3*process.qTester)
process.p = cms.Path(process.secondStep)
process.DQM.collectorHost = ''
