import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALSourceDataMonitor")

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#     #suppressInfo = cms.untracked.vstring(),
#     cout = cms.untracked.PSet(
#               threshold = cms.untracked.string('WARNING')
#           ),
#     categories = cms.untracked.vstring('*'),
#     destinations = cms.untracked.vstring('cout')
#)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load('FWCore.Modules.printContent_cfi')

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MCRUN2_74_V8A'
process.es_ascii = cms.ESSource('HcalTextCalibrations',
    input = cms.VPSet(
        cms.PSet(
            object = cms.string('ElectronicsMap'),
            file = cms.FileInPath('version_G_HF_uTCA_only_emap.txt')
            ),
	)
    )
process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# input files
process.source = cms.Source("HcalTBSource",
    fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/work/s/stepobr/public/USC_264595.root'
    )
)

# TB data unpacker
process.tbunpack = cms.EDProducer("HcalTBObjectUnpacker",
    HcalSlowDataFED = cms.untracked.int32(-1),
    HcalSourcePositionFED = cms.untracked.int32(12),
    HcalTriggerFED = cms.untracked.int32(1),
    fedRawDataCollectionTag = cms.InputTag('source')
)

process.histoUnpack = cms.EDProducer("HcalUTCAhistogramUnpacker",
          fedRawDataCollectionTag = cms.InputTag("source"))


# histo unpacker
#process.load("EventFilter.HcalRawToDigi.HcalHistogramRawToDigi_cfi")
#process.hcalhistos.HcalFirstFED = cms.untracked.int32(700)
# H2 FED is 700
#process.hcalhistos.FEDs = cms.untracked.vint32(700)
# HFM Q1, Q4: FEDS 718 and 722
#process.hcalhistos.FEDs = cms.untracked.vint32(718,722)
# HEM09,10,11 --> FEDs 706 and 708
#process.hcalhistos.FEDs = cms.untracked.vint32(706,708)

# Tree-maker
process.hcalSourceDataMon = cms.EDAnalyzer('HCALSourceDataMonitor',
    RootFileName = cms.untracked.string('hcalSourceDataMon.test.215595.root'),
    PrintRawHistograms = cms.untracked.bool(False),
    SelectDigiBasedOnTubeName = cms.untracked.bool(True),
    HcalSourcePositionDataTag = cms.InputTag("tbunpack"),
    hcalTBTriggerDataTag = cms.InputTag("tbunpack"),
    HcalUHTRhistogramDigiCollectionTag = cms.InputTag("histoUnpack"),
)

process.p = cms.Path(process.tbunpack
                     *process.histoUnpack
#		     *process.printContent
                     *process.hcalSourceDataMon
                    )

