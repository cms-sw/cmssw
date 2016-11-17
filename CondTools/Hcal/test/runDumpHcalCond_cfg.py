import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing()
options.register ("dumplist", '', VarParsing.multiplicity.list, VarParsing.varType.string)
options.register ("globaltag", '', VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("geometry", '', VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("era", '', VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("run", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("dbfile", '', VarParsing.multiplicity.singleton, VarParsing.varType.string) # 'sqlite_file:testExample.db'
options.register ("dblist", '', VarParsing.multiplicity.list, VarParsing.varType.string)
options.register ("frontierloc", 'frontier://FrontierProd/CMS_CONDITIONS', VarParsing.multiplicity.singleton, VarParsing.varType.string) # 'frontier://FrontierDev/CMS_COND_HCAL'
options.register ("frontierlist", '', VarParsing.multiplicity.list, VarParsing.varType.string)
options.register ("usehardcode", '', VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("asciilist", '', VarParsing.multiplicity.list, VarParsing.varType.string)
options.register ("prefix", 'DumpCond', VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("command", '', VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("info", '', VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("dump", '', VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.parseArguments()

allconds = [
    'Pedestals',
    'PedestalWidths',
    'Gains',
    'QIEData',
    'QIETypes',
    'ElectronicsMap',
    'ChannelQuality',
    'GainWidths',
    'RespCorrs',
    'TimeCorrs',
    'LUTCorrs',
    'PFCorrs',
    'L1TriggerObjects',
    'ZSThresholds',
    'ValidationCorrs',
    'LutMetadata',
    'DcsValues',
    'DcsMap',
    'TimingParams',
    'RecoParams',
    'LongRecoParams',
    'MCParams',
    'FlagHFDigiTimeParams',
    'SiPMParameters',
    'SiPMCharacteristics',
    'TPParameters',
    'TPChannelParameters',
    'FrontEndMap',
    'CalibrationsSet',
    'CalibrationWidthsSet',
]

#custom help message
if options.info:
    print "dumplist possibilities:"
    print allconds
    print "dbfile format: sqlite_file:foo.db"
    print "frontierloc possibilities: frontier://FrontierProd/CMS_CONDITIONS (default), frontier://FrontierDev/CMS_COND_HCAL, etc."
    print "dblist/frontierlist entry format: HcalPedestalsRcd:hcal_pedestals_fC_v6_mc"
    print "asciilist entry format: Pedestals:CondFormats/HcalObjects/data/hcal_pedestals_fC_v5.txt"
    print "command can be used to execute extra settings, newline separated, e.g.: process.es_hardcode.useHEUpgrade=cms.bool(True)\\nprocess.es_hardcode.useHFUpgrade=cms.bool(True)"
    print "dump will do the equivalent of edmConfigDump: use with python instead of cmsRun"
    print "specifying globaltag without the proper geometry may cause errors"
    
    sys.exit(0)

if not options.dumplist:
    print "Nothing to do!"
    sys.exit(0)
    
process = cms.Process("DUMP")
if options.era:
    from Configuration.StandardSequences.Eras import eras
    process = cms.Process("DUMP",getattr(eras,options.era))

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration.StandardSequences.Services_cff')

if options.globaltag:
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
    from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, options.globaltag, '')

# extracted from Configuration/Applications/python/ConfigBuilder.py
if options.geometry:
    SimGeometryCFF=''
    GeometryCFF='Configuration/StandardSequences/GeometryRecoDB_cff'
    geometryDBLabel=None
    simGeometry=''
    def inGeometryKeys(opt):
        from Configuration.StandardSequences.GeometryConf import GeometryConf
        if opt in GeometryConf:
            return GeometryConf[opt]
        else:
            return opt

    geoms=options.geometry.split(',')
    if len(geoms)==1: geoms=inGeometryKeys(geoms[0]).split(',')
    if len(geoms)==2:
        #may specify the reco geometry
        if '/' in geoms[1] or '_cff' in geoms[1]:
            GeometryCFF=geoms[1]
        else:
            GeometryCFF='Configuration/Geometry/Geometry'+geoms[1]+'_cff'

    if (geoms[0].startswith('DB:')):
        SimGeometryCFF='Configuration/StandardSequences/GeometrySimDB_cff'
        geometryDBLabel=geoms[0][3:]
    else:
        if '/' in geoms[0] or '_cff' in geoms[0]:
            self.SimGeometryCFF=geoms[0]
        else:
            simGeometry=geoms[0]
            self.SimGeometryCFF='Configuration/Geometry/Geometry'+geoms[0]+'_cff'

    if SimGeometryCFF: process.load(SimGeometryCFF)
    process.load(GeometryCFF)
    if geometryDBLabel:
        process.XMLFromDBSource.label = cms.string(geometryDBLabel)

process.prod = cms.EDAnalyzer("HcalDumpConditions",
    dump = cms.untracked.vstring(options.dumplist),
    outFilePrefix = cms.untracked.string(options.prefix)
)

# specify for which run you would like to get the conditions in the "firstRun"
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(options.run)
)

if options.dbfile and options.dblist:
    process.es_dbfile = cms.ESSource("PoolDBESSource",
        process.CondDBSetup,
        timetype = cms.string('runnumber'),
        connect = cms.string(options.dbfile),
        authenticationMethod = cms.untracked.uint32(0),
        toGet = cms.VPSet()
    )
    for rcd in options.dblist:
        rcds = rcd.split(':')
        if len(rcds) != 2: continue
        process.es_dbfile.toGet.append(cms.PSet(record = cms.string(rcds[0]), tag = cms.string(rcds[1])))
    process.es_prefer_dbfile = cms.ESPrefer('PoolDBESSource','es_dbfile')

if options.frontierloc and options.frontierlist:
    process.es_frontier = cms.ESSource("PoolDBESSource",
        process.CondDBSetup,
        timetype = cms.string('runnumber'),
        connect = cms.string(options.frontierloc),
        authenticationMethod = cms.untracked.uint32(0),
        toGet = cms.VPSet()
    )
    for rcd in options.frontierlist:
        rcds = rcd.split(':')
        if len(rcds) != 2: continue
        process.es_frontier.toGet.append(cms.PSet(record = cms.string(rcds[0]), tag = cms.string(rcds[1])))
    process.es_prefer_frontier = cms.ESPrefer('PoolDBESSource','es_frontier')

if options.asciilist:
    process.es_ascii = cms.ESSource("HcalTextCalibrations",
        input = cms.VPSet()
    )
    for obj in options.asciilist:
        objs = obj.split(':')
        if len(objs) != 2: continue
        process.es_ascii.input.append(cms.PSet(object = cms.string(objs[0]), file = cms.FileInPath(objs[1])))
    process.es_prefer_ascii = cms.ESPrefer('HcalTextCalibrations','es_ascii')

if options.usehardcode:
    # loads params and es_prefer
    process.load("CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff")
    process.es_hardcode.toGet = cms.untracked.vstring(options.dumplist)

if options.command:
    cmds = command.split('\n')
    for cmd in cmds:
        exec(cmd)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

if options.dump:
    print process.dumpPython()
    sys.exit(0)
else:
    process.p = cms.Path(process.prod)
