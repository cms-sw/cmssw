from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys, os
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate
from Configuration.Geometry.dictRun4Geometry import detectorVersionDict

varType = Enumerate ("Run1 2015 2017 2021 Run4 MaPSA")
defaultVersion=str();

def help():
   print("Usage: cmsRun dumpFWRecoGeometry_cfg.py  tag=TAG ")
   print("   tag=tagname")
   print("       identify geometry condition database tag")
   print("      ", varType.keys())
   print("")
   print("   version=versionNumber")
   print("       scenario version from Run4 dictionary")
   print("")
   print("   tgeo=bool")
   print("       dump in TGeo format to browse in geometry viewer")
   print("       import this in Fireworks with option --sim-geom-file")
   print("")
   print("   tracker=bool")
   print("       include Tracker subdetectors")
   print("")
   print("   muon=bool")
   print("       include Muon subdetectors")
   print("")
   print("   calo=bool")
   print("       include Calo subdetectors")
   print("")
   print("   timing=bool")
   print("       include Timing subdetectors")
   print("")
   print("")
   os._exit(1);

def versionCheck(ver):
   if ver == "":
      print("Please, specify Run4 scenario version\n")
      print(sorted([x[1] for x in detectorVersionDict.items()]))
      print("")
      help()

def recoGeoLoad(score, properties):
    print("Loading configuration for tag ", options.tag ,"...\n")

    if score == "Run1":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['run1_mc']
       process.load("Configuration.StandardSequences.GeometryDB_cff")
       
    elif score == "2015":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['run2_mc']
       process.load("Configuration.StandardSequences.GeometryDB_cff")
       
    elif score == "2017":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['upgrade2017']
       process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
       
    elif score == "2021":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['upgrade2021']
       ## NOTE: There is no PTrackerParameters Rcd in this GT yet
       process.load('Geometry.TrackerGeometryBuilder.trackerParameters_cfi')
       process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
       ## NOTE: There are no Muon alignement records in the GT yet
       process.DTGeometryESModule.applyAlignment = cms.bool(False)
       process.CSCGeometryESModule.applyAlignment = cms.bool(False)
       
    elif "Run4" in score:
       versionCheck(options.version)
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

       # Import the required configuration from the CMSSW module
       from Configuration.AlCa.autoCond import autoCond  # Ensure autoCond is imported

       # Ensure options.version is defined and set correctly
       version_key = 'Run4' + options.version  # This constructs the key for accessing the properties dictionary
       print(f"Constructed version key: {version_key}")

       # Check if the key exists in properties for Run4
       if version_key in properties['Run4']:
          # Get the specific global tag for this version
          global_tag_key = properties['Run4'][version_key]['GT']
          print(f"Global tag key from properties: {global_tag_key}")

          # Check if this key exists in autoCond
          if global_tag_key.replace("auto:", "") in autoCond:
             # Set the global tag
             from Configuration.AlCa.GlobalTag import GlobalTag
             process.GlobalTag = GlobalTag(process.GlobalTag, global_tag_key, '')
          else:
             raise KeyError(f"Global tag key '{global_tag_key}' not found in autoCond.")
       else:
          raise KeyError(f"Version key '{version_key}' not found in properties['Run4'].")
       process.load('Configuration.Geometry.GeometryExtendedRun4'+options.version+'Reco_cff')
       process.trackerGeometry.applyAlignment = cms.bool(False)

    elif score == "MaPSA":
       process.load('Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff')
       process.load('Geometry.TrackerCommonData.mapsaGeometryXML_cfi')
       process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
       process.load('Geometry.TrackerNumberingBuilder.trackerTopology_cfi')
       process.load('Geometry.TrackerGeometryBuilder.trackerParameters_cfi')
       process.load('Geometry.TrackerGeometryBuilder.trackerGeometry_cfi')
       process.trackerGeometry.applyAlignment = cms.bool(False)
       process.load('RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi')

       process.load('Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi')
       
    elif score == "HGCTB160": ## hgcal testbeam
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") 
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']
       process.load('Geometry.HGCalTBCommonData.hgcalTBParametersInitialization_cfi')
       process.load('Geometry.HGCalTBCommonData.hgcalTBNumberingInitialization_cfi')
       process.load('Geometry.CaloEventSetup.HGCalTBTopology_cfi')
       process.load('Geometry.HGCalGeometry.HGCalTBGeometryESProducer_cfi')
       process.load('Geometry.CaloEventSetup.CaloTopology_cfi')
       process.load('Geometry.CaloEventSetup.CaloGeometryBuilder_cfi')
       process.CaloGeometryBuilder = cms.ESProducer(
          "CaloGeometryBuilder",
          SelectedCalos = cms.vstring("HGCalEESensitive")
       )
       process.load("SimG4CMS.HGCalTestBeam.HGCalTB160XML_cfi")
       
    else:
      help()


options = VarParsing.VarParsing ()

defaultOutputFileName="cmsRecoGeom.root"

options.register ('tag',
                  "2017", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "tag info about geometry database conditions")

options.register ('version',
                  defaultVersion, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "info about Run4 geometry scenario version")

options.register ('tgeo',
                  False, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "write geometry in TGeo format")

options.register ('tracker',
                  True, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "write Tracker geometry")

options.register ('muon',
                  True, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "write Muon geometry")

options.register ('calo',
                  True, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "write Calo geometry")

options.register ('timing',
                  False, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "write Timing geometry")

options.register ('out',
                  defaultOutputFileName, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Output file name")

options.parseArguments()

from Configuration.PyReleaseValidation.upgradeWorkflowComponents import upgradeProperties as properties
# Determine version_key based on the value of options.tag
if options.tag == "Run4" or options.tag == "MaPSA":
   prop_key = 'Run4'
   version_key = options.tag + options.version
elif options.tag == "2017" or options.tag == "2021": #(this leads to crashes in tests ?)
   prop_key = 2017
   version_key = options.tag
else:
   prop_key = None
   version_key = None

if(prop_key and version_key):
   print(f"Constructed version key: {version_key}")
   era_key = properties[prop_key][str(version_key)]['Era']
   print(f"Constructed era key: {era_key}")
   from Configuration.StandardSequences.Eras import eras
   era = getattr(eras, era_key)
   process = cms.Process("DUMP",era)
else:
   process = cms.Process("DUMP")
process.add_(cms.Service("InitRootHandlers", ResetRootErrHandler = cms.untracked.bool(False)))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


recoGeoLoad(options.tag,properties)

if ( options.tgeo == True):
    if (options.out == defaultOutputFileName ):
       options.out = "cmsTGeoRecoGeom-" + str(options.tag) + (f"_{options.version}" if options.version else "") + ".root"
    process.add_(cms.ESProducer("FWTGeoRecoGeometryESProducer",
                 Tracker = cms.untracked.bool(options.tracker),
                 Muon = cms.untracked.bool(options.muon),
                 Calo = cms.untracked.bool(options.calo),
                 Timing = cms.untracked.bool(options.timing)))
    process.dump = cms.EDAnalyzer("DumpFWTGeoRecoGeometry",
                              tagInfo = cms.untracked.string(options.tag),
                       outputFileName = cms.untracked.string(options.out)
                              )
else:
    if (options.out == defaultOutputFileName ):
       options.out = "cmsRecoGeom-" + str(options.tag) + (f"_{options.version}" if options.version else "") + ".root"
    process.add_(cms.ESProducer("FWRecoGeometryESProducer",
                 Tracker = cms.untracked.bool(options.tracker),
                 Muon = cms.untracked.bool(options.muon),
                 Calo = cms.untracked.bool(options.calo),
                 Timing = cms.untracked.bool(options.timing)))
    process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                              level   = cms.untracked.int32(1),
                              tagInfo = cms.untracked.string(options.tag),
                       outputFileName = cms.untracked.string(options.out)
                              )

print("Dumping geometry in " , options.out, "\n"); 
process.p = cms.Path(process.dump)


