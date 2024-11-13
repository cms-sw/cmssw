from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys, os
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate
from Configuration.Geometry.dictRun4Geometry import detectorVersionDict

varType = Enumerate ("Run1 2015 2015dev 2017 2017Muon 2021 Run4")
defaultVersion=str();

def help():
   print("Usage: cmsRun dumpSimGeometry_cfg.py  tag=TAG version=VERSION ")
   print("   tag=tagname")
   print("       identify geometry scenario ")
   print("      ", varType.keys())
   print("")
   print("   version=versionNumber")
   print("       scenario version from Run4 dictionary")
   print("")
   print("   out=outputFileName")
   print("       default is cmsSimGeom<tag><version>.root")
   print() 
   os._exit(1);

def versionCheck(ver):
   if ver == "":
      print("Please, specify Run4 scenario version\n")
      print(sorted([x[1] for x in detectorVersionDict.items()]))
      print("")
      help()

def simGeoLoad(score):
    print("Loading configuration for scenario", options.tag , options.version ,"...\n")
    if score == "Run1":
       process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

    elif score == "2015":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015XML_cfi")

    elif score == "2015dev":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015devXML_cfi")

    elif score == "2017":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2017XML_cfi")
       
    elif score == "2017Muon":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2017MuonXML_cfi")

    elif score == "2021":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2021XML_cfi")

    elif score == "Run4":
       versionCheck(options.version)
       process.load("Geometry.CMSCommonData.cmsExtendedGeometryRun4" + options.version + "XML_cfi")
 
    else:
      help()

options = VarParsing.VarParsing ()

defaultTag=str(2017);
defaultLevel=14;
defaultOutputFileName="cmsSimGeom-"+ defaultTag +".root"

options.register ('tag',
                  defaultTag, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "info about geometry scenario")
options.register ('version',
                  defaultVersion, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "info about Run4 geometry scenario version")
options.register ('out',
                  defaultOutputFileName, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Output file name")


options.parseArguments()


if (options.out == defaultOutputFileName ):
   options.out = "cmsSimGeom-" + str(options.tag) + str(options.version) + ".root"

process = cms.Process("SIMDUMP")
simGeoLoad(options.tag)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.add_(cms.ESProducer("TGeoMgrFromDdd",
                            verbose = cms.untracked.bool(False),
                            level = cms.untracked.int32(defaultLevel)
                            ))

process.dump = cms.EDAnalyzer("DumpSimGeometry", 
                              tag = cms.untracked.string(options.tag),
                              outputFileName = cms.untracked.string(options.out))

process.p = cms.Path(process.dump)
