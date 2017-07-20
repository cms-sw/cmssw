import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.register('GT',	        '', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('frontier',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('record',	'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, '') 
options.register('run',		'', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, '') 
options.parseArguments()

process = cms.Process("DUMP")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.Geometry.GeometryExtended2017Plan1_cff")
process.load("Configuration.Geometry.GeometryExtended2017Plan1Reco_cff")
process.load("CondCore.CondDB.CondDB_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(options.run)
)

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

if options.GT:
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    process.GlobalTag.globaltag = options.GT
else:
    process.CondDB.connect = options.frontier
    process.es_pool = cms.ESSource("PoolDBESSource",
        process.CondDB,
        timetype = cms.string('runnumber'),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string("Hcal"+options.record+"Rcd"),
                tag = cms.string(options.tag)
            )),
      authenticationMethod = cms.untracked.uint32(0),
    )

process.dumpcond = cms.EDAnalyzer("HcalDumpConditions",
       dump = cms.untracked.vstring(options.record)
)
process.p = cms.Path(process.dumpcond)

