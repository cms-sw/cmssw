import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCondVisualizer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripCondVisualizer=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripCondVisualizer = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(306054),  
                            lastValue = cms.uint64(306054),   
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

process.load('Configuration.StandardSequences.GeometryRecoDB_cff') 
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')

# quality producer needed for the bad components dumping
from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import*
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    # cms.PSet(record = cms.string("SiStripDetVOffRcd"), tag = cms.string('')),  # DCS information
    cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string('')), # Use Detector cabling information to exclude detectors not connected
    cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string('')), # Online Bad components
    cms.PSet(record = cms.string('SiStripBadFiberRcd'), tag = cms.string('')),   # Bad Channel list from the selected IOV as done at PCL
    cms.PSet(record = cms.string('RunInfoRcd'), tag = cms.string(''))            # List of FEDs exluded during data taking
)

siStripQualityESProducer.ReduceGranularity = cms.bool(False)
siStripQualityESProducer.ThresholdForReducedGranularity = cms.double(0.3)
siStripQualityESProducer.appendToDataLabel = 'MergedBadComponent'
siStripQualityESProducer.PrintDebugOutput = cms.bool(True)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("conditionsDump.root"))

from CondTools.SiStrip.siStripCondVisualizer_cfi import siStripCondVisualizer
process.condVisualizer = siStripCondVisualizer.clone(doNoise = True,
                                                     doPeds = True,
                                                     doG1 = True,
                                                     doG2 = True,
                                                     doBadComps = True,
                                                     # example of configuration of DetId selector
                                                     # selections=cms.VPSet(
                                                     #     cms.PSet(detSelection = cms.uint32(30),detLabel = cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),      # TIB
                                                     #     cms.PSet(detSelection = cms.uint32(41),detLabel = cms.string("TIDMinus"),selection=cms.untracked.vstring("0x1e006000-0x18002000")), # TID minus
                                                     #     cms.PSet(detSelection = cms.uint32(42),detLabel = cms.string("TIDPlus"),selection=cms.untracked.vstring("0x1e006000-0x18004000")),  # TID plus
                                                     #     cms.PSet(detSelection = cms.uint32(50),detLabel = cms.string("TOB"),selection=cms.untracked.vstring("0x1e000000-0x1a000000")),      # TOB
                                                     #     cms.PSet(detSelection = cms.uint32(61),detLabel = cms.string("TECMinus"),selection=cms.untracked.vstring("0x1e0c0000-0x1c040000")), # TEC minus
                                                     #     cms.PSet(detSelection = cms.uint32(62),detLabel = cms.string("TECPlus"),selection=cms.untracked.vstring("0x1e0c0000-0x1c080000"))   # TEC plus
                                                     # )
                                                    )
process.p = cms.Path(process.condVisualizer)
