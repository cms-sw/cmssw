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


##-- GT conditions for all
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['com10'] ## == GR_R_53_V16::All in 5_3_7
###-- Customized particular conditions
from CondCore.DBCommon.CondDBSetup_cfi import *

# EMAP NEEDED FOR H2 DATA
##----------------------------------- replacing conditions with txt ones
#process.es_ascii = cms.ESSource("HcalTextCalibrations",
#    input = cms.VPSet(
#      cms.PSet(
#        object = cms.string('ElectronicsMap'),
#        file = cms.FileInPath('HCALSourcing/HCALSourceDataMonitor/emap_HCAL_H2_BI_modSIC_nov2013.txt')
#        )
#      )
#    )
#process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# input files
process.source = cms.Source("HcalTBSource",
    fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/user/s/stepobr/eos/cms/store/group/dpg_hcal/comm_hcal/LS1/USC_221451.root'
    )
)

# TB data unpacker
process.tbunpack = cms.EDProducer("HcalTBObjectUnpacker",
    HcalSlowDataFED = cms.untracked.int32(-1),
    HcalSourcePositionFED = cms.untracked.int32(12),
    HcalTriggerFED = cms.untracked.int32(1),
    fedRawDataCollectionTag = cms.InputTag('rawDataCollector')
)

# histo unpacker
process.load("EventFilter.HcalRawToDigi.HcalHistogramRawToDigi_cfi")
process.hcalhistos.HcalFirstFED = cms.untracked.int32(700)
# H2 FED is 700
#process.hcalhistos.FEDs = cms.untracked.vint32(700)
# HFM Q1, Q4: FEDS 718 and 722
#process.hcalhistos.FEDs = cms.untracked.vint32(718,722)
# HEM09,10,11 --> FEDs 706 and 708
#process.hcalhistos.FEDs = cms.untracked.vint32(706,708)
#process.hcalhistos.FEDs = cms.untracked.vint32(719,721,723)
#process.hcalhistos.FEDs = cms.untracked.vint32(718, 719, 720, 721, 722, 723)
process.hcalhistos.FEDs = cms.untracked.vint32(718, 719, 720, 721, 722, 723)

# Tree-maker
process.hcalSourceDataMon = cms.EDAnalyzer('HCALSourceDataMonitor',
    RootFileName = cms.untracked.string('221451_hcalSourceDataMon.root'),
    PrintRawHistograms = cms.untracked.bool(False),
    SelectDigiBasedOnTubeName = cms.untracked.bool(True)
)

process.p = cms.Path(process.tbunpack
                     *process.hcalhistos
                     *process.hcalSourceDataMon
                    )

