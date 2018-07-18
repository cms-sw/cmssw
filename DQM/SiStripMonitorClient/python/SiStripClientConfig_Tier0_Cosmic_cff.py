import FWCore.ParameterSet.Config as cms

#  SiStripOfflineDQM (for Tier0 Harvesting Step) ####
siStripOfflineAnalyser = cms.EDAnalyzer("SiStripOfflineDQM",
    GlobalStatusFilling      = cms.untracked.int32(2),
    CreateSummary            = cms.untracked.bool(False),
    SummaryConfigPath        = cms.untracked.string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml"),
    UsedWithEDMtoMEConverter = cms.untracked.bool(True),
    PrintFaultyModuleList    = cms.untracked.bool(True),
    CreateTkMap              = cms.untracked.bool(False), 
    TrackRatePSet            = cms.PSet(
           Name     = cms.string("NumberOfTracks_"),
                  LowerCut = cms.double(0.0),
                  UpperCut = cms.double(100.0),
               ),
)

siStripQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0_cosmic.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)

from CalibTracker.SiStripESProducers.SiStripBadModuleFedErrESSource_cfi import*
siStripBadModuleFedErrESSource.appendToDataLabel = cms.string('BadModules_from_FEDBadChannel')
siStripBadModuleFedErrESSource.ReadFromFile = cms.bool(False)

from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import siStripQualityESProducer 
mergedSiStripQualityProducer = siStripQualityESProducer.clone(
    #names and desigantions
    ListOfRecordToMerge = cms.VPSet(
        cms.PSet(record = cms.string("SiStripDetVOffRcd"), tag = cms.string('')), # DCS information
        cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string('')), # Use Detector cabling information to exclude detectors not connected            
        cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string('')), # Online Bad components
        cms.PSet(record = cms.string('SiStripBadFiberRcd'), tag = cms.string('')),   # Bad Channel list from the selected IOV as done at PCL
        cms.PSet(record = cms.string('SiStripBadModuleFedErrRcd'), tag = cms.string('BadModules_from_FEDBadChannel')), # BadChannel list from FED erroes              
        cms.PSet(record = cms.string('RunInfoRcd'), tag = cms.string(''))            # List of FEDs exluded during data taking          
        )
    )

mergedSiStripQualityProducer.ReduceGranularity = cms.bool(False)
mergedSiStripQualityProducer.ThresholdForReducedGranularity = cms.double(0.3)
mergedSiStripQualityProducer.appendToDataLabel = 'MergedBadComponent'


siStripBadComponentInfo = cms.EDProducer("SiStripBadComponentInfo",
    StripQualityLabel = cms.string('MergedBadComponent')
)

# Sequence
SiStripCosmicDQMClient = cms.Sequence(siStripQTester*siStripOfflineAnalyser*siStripBadComponentInfo)
#removed modules using TkDetMap
#SiStripCosmicDQMClient = cms.Sequence(siStripQTester)


# Services needed for TkHistoMap
from CalibTracker.SiStripCommon.TkDetMap_cff import *
