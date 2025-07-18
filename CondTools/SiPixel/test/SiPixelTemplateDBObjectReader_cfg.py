import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as opts
import sys

options = opts.VarParsing ('standard')

options.register('MagField',
    		 None,
    		 opts.VarParsing.multiplicity.singleton,
    		 opts.VarParsing.varType.float,
    		 'Magnetic field value in Tesla')
options.register('Year',
    		 None,
    		 opts.VarParsing.multiplicity.singleton,
    		 opts.VarParsing.varType.string,
    		 'Current year for versioning')
options.register('Version',
    		 None,
    		 opts.VarParsing.multiplicity.singleton,
    		 opts.VarParsing.varType.string,
    		 'Template DB object version')
options.register('Append',
    		 None,
    		 opts.VarParsing.multiplicity.singleton,
    		 opts.VarParsing.varType.string,
    		 'Any additional string to add to the filename, i.e. "bugfix", etc.')
options.register('GlobalTag',
    		 'auto:run3_data',
    		 opts.VarParsing.multiplicity.singleton,
    		 opts.VarParsing.varType.string,
    		 'Global tag for this run')
options.register('readFromGT',
                 False,
                 opts.VarParsing.multiplicity.singleton,
    		 opts.VarParsing.varType.bool,
    		 'read template from Global Tag')
options.parseArguments()

from Configuration.StandardSequences.Eras import eras
process = cms.Process("SiPixelTemplateDBReaderTest",eras.Run3)

#Load the correct Magnetic Field
magfieldstrsplit = str(options.MagField).split('.')
MagFieldString = magfieldstrsplit[0]
if len(magfieldstrsplit)>1 :
	MagFieldString+=magfieldstrsplit[1]

process.load("Configuration.StandardSequences.MagneticField_"+MagFieldString+"T_cff")

if(not options.readFromGT):
        MagFieldValue = 10.*options.MagField #code needs it in deciTesla
        print('\nMagField = %f deciTesla \n'%(MagFieldValue))
        version = options.Version
        print('\nVersion = %s \n'%(version))
        template_base = 'SiPixelTemplateDBObject_'+MagFieldString+'T_'+options.Year+'_v'+version
        print("Testing sqlite file: "+template_base+".db")
        print("                tag: "+template_base)

process.load("CondCore.CondDB.CondDB_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')

#Change to True if you would like a more detailed error output
wantDetailedOutput = False
#Change to True if you would like to output the full template database object
wantFullOutput = False

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(386863)
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# (optionally) load the template
if not options.readFromGT:
        process.TemplateDBSource = cms.ESSource("PoolDBESSource",
                                            DBParameters = cms.PSet(
                                                    messageLevel = cms.untracked.int32(0),
                                                    authenticationPath = cms.untracked.string('')),
                                            toGet = cms.VPSet(cms.PSet(
                                                    record = cms.string('SiPixelTemplateDBObjectRcd'),
                                                    tag = cms.string(template_base))),
                                            connect = cms.string('sqlite_file:'+template_base+'.db'))
    
        process.prefer_TemplateDBSource = cms.ESPrefer("PoolDBESSource","TemplateDBSource")
else:
        process.load("CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi")
        
process.reader = cms.EDAnalyzer("SiPixelTemplateDBObjectReader",
                                siPixelTemplateCalibrationLocation = cms.string("CalibTracker/SiPixelESProducers"),
                                wantDetailedTemplateDBErrorOutput = cms.bool(wantDetailedOutput),
                                wantFullTemplateDBOutput = cms.bool(wantFullOutput),
                                TestGlobalTag = cms.bool(options.readFromGT)
                                )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.reader)
