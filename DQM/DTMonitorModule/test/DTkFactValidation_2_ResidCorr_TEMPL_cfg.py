import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
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
        resolution = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('resolution'),
    destinations = cms.untracked.vstring('cout')
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests_ttrig.xml')
)

#process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
#process.modulo1=process.resolutionTest.clone(
#  histoTag2D = 'hResDistVsDist_STEP1', 
#  histoTag  = 'hResDist_STEP1',
#  STEP = 'STEP1',
#  OutputMEsInRootFile = False,
#  readFile = True,
#  inputFile = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/DTkFactValidation_ResidCorr_RUNNUMBERTEMPLATE.root'
# )
 
#process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
#process.modulo2=process.resolutionTest.clone(
#  histoTag2D = 'hResDistVsDist_STEP2', 
#  histoTag  = 'hResDist_STEP2',
#  STEP = 'STEP2',
#  OutputMEsInRootFile = False,
#  readFile = True,
#  inputFile = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/DTkFactValidation_ResidCorr_RUNNUMBERTEMPLATE.root'
# )
 
process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.modulo=process.resolutionTest.clone(
  histoTag2D = 'hResDistVsDist_STEP3',
  histoTag  = 'hResDist_STEP3',
  STEP = 'STEP3',
  OutputMEsInRootFile = True,
  readFile = True,
  inputFile = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/DTkFactValidation_ResidCorr_RUNNUMBERTEMPLATE.root',
  OutputFileName = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/SummaryResiduals_ResidCorr_RUNNUMBERTEMPLATE.root'
)

process.secondStep = cms.Sequence(process.modulo*process.qTester)
process.p = cms.Path(process.secondStep)
process.DQM.collectorHost = ''
