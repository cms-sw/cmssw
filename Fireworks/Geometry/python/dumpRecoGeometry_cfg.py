from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys, os
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate
from Configuration.Geometry.dict2026Geometry import detectorVersionDict

varType = Enumerate ("Run1 2015 2017 2021 2026 MaPSA")
defaultVersion=str();

def help():
   print("Usage: cmsRun dumpFWRecoGeometry_cfg.py  tag=TAG ")
   print("   tag=tagname")
   print("       identify geometry condition database tag")
   print("      ", varType.keys())
   print("")
   print("   version=versionNumber")
   print("       scenario version from 2026 dictionary")
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
      print("Please, specify 2026 scenario version\n")
      print(sorted([x[1] for x in detectorVersionDict.items()]))
      print("")
      help()

def recoGeoLoad(score):
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
       
    elif  score == "2021":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['upgrade2021']
       ## NOTE: There is no PTrackerParameters Rcd in this GT yet
       process.load('Geometry.TrackerGeometryBuilder.trackerParameters_cfi')
       process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
       ## NOTE: There are no Muon alignement records in the GT yet
       process.DTGeometryESModule.applyAlignment = cms.bool(False)
       process.CSCGeometryESModule.applyAlignment = cms.bool(False)
       
    elif "2026" in score:
       versionCheck(options.version)
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['run2_mc']
       process.load('Configuration.Geometry.GeometryExtended2026'+options.version+'Reco_cff')
       
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
       process.load('Geometry.HGCalCommonData.hgcalV6ParametersInitialization_cfi')
       process.load('Geometry.HGCalCommonData.hgcalV6NumberingInitialization_cfi')
       process.load('Geometry.CaloEventSetup.HGCalV6Topology_cfi')
       process.load('Geometry.HGCalGeometry.HGCalV6GeometryESProducer_cfi')
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
                  "info about 2026 geometry scenario version")

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




process = cms.Process("DUMP")
process.add_(cms.Service("InitRootHandlers", ResetRootErrHandler = cms.untracked.bool(False)))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


recoGeoLoad(options.tag)

if ( options.tgeo == True):
    if (options.out == defaultOutputFileName ):
        options.out = "cmsTGeoRecoGeom-" +  str(options.tag) + ".root"
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
        options.out = "cmsRecoGeom-" +  str(options.tag) + ".root"
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


