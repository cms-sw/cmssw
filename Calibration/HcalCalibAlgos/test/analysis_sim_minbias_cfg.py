import FWCore.ParameterSet.Config as cms

process = cms.Process('SimAna')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('RecoLocalCalo.Configuration.RecoLocalCalo_cff')
process.load('RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff')
process.load('Geometry.HcalEventSetup.HcalTopology_cfi')
process.load('Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff')


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.163 $'),
    annotation = cms.untracked.string('Reconstruction.py nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
    )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )
process.options = cms.untracked.PSet(
    
    )

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
     'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/559/FA519929-93C0-E111-9C4A-BCAEC532971C.root',
    )
)

############################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup'] #data

process.es_pool = cms.ESSource("PoolDBESSource",
                               process.CondDBSetup,
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(
    cms.PSet(
    record = cms.string("HcalRespCorrsRcd"),
    tag = cms.string("HcalRespCorrs_v4.5_offline")),
    
    cms.PSet(
    record = cms.string('HcalGainsRcd'),
    tag = cms.string('HcalGains_v5.07_offline')),
    
    ),
                               connect = cms.string('frontier://FrontierProd/CMS_COND_31X_HCAL'),
                               authenticationMethod = cms.untracked.uint32(0),
                               )
process.es_prefer_es_pool = cms.ESPrefer( "PoolDBESSource", "es_pool")

############################################################################
#### Analysis

process.minbiasana = cms.EDAnalyzer("SimAnalyzerMinbias",
                                    HistOutFile = cms.untracked.string('simanalyzer.root'),
                                    TimeCut = cms.untracked.double(100.0)
                                    )

process.schedule = cms.Path(    
    process.minbiasana*
    process.endOfProcess   
    )


