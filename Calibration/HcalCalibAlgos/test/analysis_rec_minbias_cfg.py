import FWCore.ParameterSet.Config as cms

process = cms.Process('RecAna')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Geometry.HcalEventSetup.HcalTopology_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('RecoLocalCalo.Configuration.RecoLocalCalo_cff')
process.load('RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff')
process.load('Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff')


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
   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/203/522/F233228C-D206-E211-BC2E-0025901D629C.root',
   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/556/D673F957-93C0-E111-8332-003048D2BF1C.root',
   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/198/941/3AA08F69-9ECD-E111-A338-5404A63886C5.root',
   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/601/BAC19C2C-3EC1-E111-8A78-001D09F241B9.root',
    'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/610/F68235DD-01C1-E111-9583-0019B9F72CE5.root',
   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/719/2AADB426-7BC1-E111-9053-001D09F24353.root', 
    'root://cmsxrootd.fnal.gov//store/data/Run2012A/HcalNZS/RAW/v1/000/190/456/88BBA906-3B7F-E111-BD13-001D09F2441B.root',
    'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/199/975/24F29491-FADA-E111-A19B-BCAEC5329716.root',
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

process.minbiasana = cms.EDAnalyzer("Analyzer_minbias",
                                    HistOutFile = cms.untracked.string('recanalyzer.root'),
                                    hbheInputMB = cms.InputTag("hbherecoMB"),
                                    hoInputMB = cms.InputTag("horecoMB"),
                                    hfInputMB = cms.InputTag("hfrecoMBspecial"),
                                    hbheInputNoise = cms.InputTag("hbherecoNoise"),
                                    hoInputNoise = cms.InputTag("horecoNoise"),
                                    hfInputNoise = cms.InputTag("hfrecoNoise"),
                                    triglabel=cms.untracked.InputTag('TriggerResults','','HLT'),
                                    Recalib = cms.bool(False)
                                    )

process.schedule = cms.Path(    
    process.hcalDigis*
    process.gtDigis*
    process.hcalLocalRecoSequenceNZS*
    process.seqALCARECOHcalCalMinBiasNoHLT*
    process.minbiasana*
    process.endOfProcess   
    )



