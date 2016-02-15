import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALSourceDataMonitor")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

##-- GT conditions for all
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['com10'] ## == GR_R_53_V16::All in 5_3_7
###-- Customized particular conditions
from CondCore.DBCommon.CondDBSetup_cfi import *


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("HcalTBSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
XXX_FILENAMES_XXX
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
process.hcalhistos.FEDs = cms.untracked.vint32(706,708)

# Tree-maker
process.hcalSourceDataMon = cms.EDAnalyzer('HCALSourceDataMonitor',
    RootFileName = cms.untracked.string('XXX_TFILENAME_XXX'),
    PrintRawHistograms = cms.untracked.bool(False),
    SelectDigiBasedOnTubeName = cms.untracked.bool(True)
)

process.p = cms.Path(process.tbunpack
                     *process.hcalhistos
                     *process.hcalSourceDataMon
                    )

