import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Test")

options = VarParsing.VarParsing("analysis")

options.register ('isUnitTest',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.bool,            # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  359334,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,            # string, int, or float
                  "run number")

options.register ('FEDsToAdd',
                  [],
                  VarParsing.VarParsing.multiplicity.list,  # singleton or list
                  VarParsing.VarParsing.varType.int,        # string, int, or float
                  "list of FEDs to Add")

options.register ('FEDsToRemove',
                  [],
                  VarParsing.VarParsing.multiplicity.list,  # singleton or list
                  VarParsing.VarParsing.varType.int,        # string, int, or float
                  "list of FEDs to Remove")

options.register ('DetIdsToAdd',
                  [],
                  VarParsing.VarParsing.multiplicity.list,  # singleton or list
                  VarParsing.VarParsing.varType.int,        # string, int, or float
                  "list of DetIds to Add")

options.register ('DetIdsToRemove',
                  [],
                  VarParsing.VarParsing.multiplicity.list,  # singleton or list
                  VarParsing.VarParsing.varType.int,        # string, int, or float
                  "list of DetIds to Remove")


options.register ('inputConnection',
                  "frontier://FrontierProd/CMS_CONDITIONS",
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,            # string, int, or float
                  "input DB connection")

options.register ('inputTag',
                  "SiStripBadChannel_Ideal_31X_v2",
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,            # string, int, or float
                  "input DB tag")

options.register ('outputTag',
                  "output",
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,            # string, int, or float
                  "output DB tag")


options.parseArguments()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING') if options.isUnitTest else cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

##
## Database services
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Neeeded for the cabling
##
process.CondDB.connect='frontier://FrontierProd/CMS_CONDITIONS'
process.CablingESSource = cms.ESSource('PoolDBESSource',
                                       process.CondDB,
                                       BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                       toGet = cms.VPSet( cms.PSet(record = cms.string('SiStripFedCablingRcd'),
                                                                   tag    = cms.string('SiStripFedCabling_GR10_v1_hlt')   # real data cabling map
                                                                   #tag     = cms.string('SiStripFedCabling_Ideal_31X_v2')  # ideal cabling map
                                                               )))

process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")
process.load("CalibTracker.SiStripESProducers.SiStripConnectivity_cfi")

##
## Input bad components
##
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      DBParameters = cms.PSet(
                                          messageLevel = cms.untracked.int32(2),
                                          authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                      ),
                                      timetype = cms.string('runnumber'),
                                      toGet = cms.VPSet(cms.PSet(
                                          record = cms.string('SiStripBadStripRcd'),
                                          tag = cms.string(options.inputTag)
                                      )),
                                      connect = cms.string(options.inputConnection)
                                      )

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:outputDB.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripBadStripRcd'),
                                                                     tag = cms.string(options.outputTag)
                                                                     )
                                                            )
                                          )

process.prod = cms.EDAnalyzer("SiStripBadChannelPatcher",
                              printDebug = cms.bool(True) if options.isUnitTest else cms.bool(False),
                              Record = cms.string("SiStripBadStripRcd"),
                              FEDsToExclude = cms.vuint32(options.FEDsToRemove),
                              FEDsToInclude = cms.vuint32(options.FEDsToAdd),
                              detIdsToExclude = cms.vuint32(options.DetIdsToAdd),
                              detIdsToInclude = cms.vuint32(options.DetIdsToRemove))

process.Timing = cms.Service("Timing")
process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)
