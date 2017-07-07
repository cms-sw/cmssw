import FWCore.ParameterSet.Config as cms
process = cms.Process('ANALYSIS')

process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']
#process.GlobalTag.globaltag = 'START53_V15::All'

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('warnings','errors',
                                         'cout','cerr'),
    categories = cms.untracked.vstring('HcalIsoTrack'), 
    debugModules = cms.untracked.vstring('*'),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(500),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
     ),
     cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalIsoTrack = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
       )
    )
)

process.load('Calibration.HcalCalibAlgos.HcalIsoTrkAnalyzer_cfi')
process.HcalIsoTrkAnalyzer.ProcessName  = 'HLTNew1'
process.HcalIsoTrkAnalyzer.ProducerName = 'ALCAISOTRACK'
process.HcalIsoTrkAnalyzer.ModuleName   = 'IsoProd'
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/work/g/gwalia/calib/dqm/test_29_04/CMSSW_7_5_0_pre2/src/Calibration/HcalCalibAlgos/test/PoolOutput.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output_alca.root')
)
process.p = cms.Path(process.HcalIsoTrkAnalyzer)

