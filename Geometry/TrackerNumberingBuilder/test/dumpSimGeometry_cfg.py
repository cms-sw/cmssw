from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 Ideal2015 Ideal2015dev 2015 2015dev GEMDev RPC4RE11 2017 2021 2023 2023dev 2023sim 2023Muon MaPSA CRack DB")

def help():
   print("Usage: cmsRun dumpSimGeometry_cfg.py  tag=TAG ")
   print("   tag=tagname")
   print("       indentify geometry condition database tag")
   print("      ", varType.keys())
   print("")
   print("   out=outputFileName")
   print("       default is cmsSimGeom<tag>.root")
   print() 
   exit(1);

def simGeoLoad(score):
    print("Loading configuration for tag ", options.tag ,"...\n")
    if score == "Run1":
       process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

    elif score == "2015":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015XML_cfi")

    elif score == "2015dev":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015devXML_cfi")

    elif score == "GEMDev":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015MuonGEMDevXML_cfi")
       
    elif score == "Ideal2015":
       process.load("Geometry.CMSCommonData.cmsIdealGeometry2015XML_cfi")

    elif score == "Ideal2015dev":
       process.load("Geometry.CMSCommonData.cmsIdealGeometry2015devXML_cfi")

    elif score == "RPC4RE11":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015XML_RPC4RE11_cfi")

    elif score == "2017":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2017XML_cfi')
       
    elif score == "2021":
       process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
  
    elif score == "2023dev":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2023devXML_cfi')

    elif score == "2023sim":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2023simXML_cfi')
 
    elif score == "2023Muon":
       process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')

    elif score == "2023":
       process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')

    elif score == "MaPSA":
       process.load('Geometry.TrackerCommonData.mapsaGeometryXML_cfi')

    elif score == "CRack":
       process.load('Geometry.TrackerCommonData.crackGeometryXML_cfi')

    elif score == "DB":
        process.load("Configuration.StandardSequences.GeometryDB_cff")
        process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
        from Configuration.AlCa.autoCond import autoCond
        process.GlobalTag.globaltag = autoCond['run2_mc']

    else:
      help()



options = VarParsing.VarParsing ()

defaultTag=str(2015);
defaultLevel=14;
defaultOutputFileName="cmsSimGeom.root"

options.register ('tag',
                  defaultTag, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "info about geometry database conditions")
options.register ('out',
                  defaultOutputFileName, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Output file name")


options.parseArguments()


if (options.out == defaultOutputFileName ):
   options.out = "cmsSimGeom-" + str(options.tag) + ".root"

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
