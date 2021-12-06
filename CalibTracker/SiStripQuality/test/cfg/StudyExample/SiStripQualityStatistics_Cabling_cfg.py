import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("SiStripQualityStatisticsCabling")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('cablingTagName',
                  "SiStripFedCabling_GR10_v1_hlt",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Cabling DB tag name")
options.register ('runNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.parseArguments()

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    log_cabling = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
        SiStripQualityStatistics = cms.untracked.PSet(limit=cms.untracked.int32(100000))
    ),
    destinations = cms.untracked.vstring('log_cabling','cout'),
    categories = cms.untracked.vstring('SiStripQualityStatistics')
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue= cms.uint64(options.runNumber),
    lastValue= cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#-------------------------------------------------
# Calibration
#-------------------------------------------------

process.load("Configuration.Geometry.GeometryIdeal_cff")   # needed because the GlobalTag is NOT used

process.load("CondCore.DBCommon.CondDBCommon_cfi")   # needed because the GlobalTag is NOT used
process.CondDBCommon.connect='frontier://FrontierProd/CMS_CONDITIONS'
process.poolDBESSource=cms.ESSource("PoolDBESSource",
                                    process.CondDBCommon,
                                    BlobStreamerName=cms.untracked.string('TBufferBlobStreamingService'),
                                    toGet           =cms.VPSet(
    cms.PSet(
    record=cms.string('SiStripFedCablingRcd'),
    tag   =cms.string(options.cablingTagName)
    )
    )
                                    )
    
#process.load("CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff")                                    )
process.sistripconn = cms.ESProducer("SiStripConnectivity")  # needed because the GlobalTag is NOT used

# Include masking #

process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
    cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string(''))
    )
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)


#-------------------------------------------------
# Services for the TkHistoMap
#-------------------------------------------------
process.load("DQM.SiStripCommon.TkHistoMap_cff")

from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.stat = siStripQualityStatistics.clone(
        TkMapFileName=cms.untracked.string("TkMapBadComponents_Cabling.png")  #available filetypes: .pdf .png .jpg .svg
        )


process.p = cms.Path(process.stat)


