import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 Run2 Run3 2015 2017 2019 2023D1 2023D2 2023D3 2023D4 2023D5")

def help():
   print "Usage: cmsRun dumpSimGeometry_cfg.py  tag=TAG "
   print "   tag=tagname"
   print "       indentify geometry condition database tag"
   print "      ", varType.keys()
   print ""
   print "   out=outputFileName"
   print "       default is cmsSimGeom<tag>.root"
   print 
   exit(1);

def simGeoLoad(score):
    print "Loading configuration for tag ", options.tag ,"...\n"
    if score == "Run1":
       process.load('Configuration.StandardSequences.GeometryDB_cff')
       process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['run1_mc']

    elif score == "Run2":
       process.load('Configuration.StandardSequences.GeometryDB_cff')
       process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['run2_mc']
       
    elif score == "Run3":
       process.load('Configuration.StandardSequences.GeometryDB_cff')
       process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['upgrade2017']

    elif score == "2015":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2015XML_cfi')

    elif score == "2017":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2017XML_cfi')
       
    elif score == "2019":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2019XML_cfi')
  
    elif score == "2023D1":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2023D1XML_cfi')

    elif score == "2023D2":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2023D2XML_cfi')
 
    elif score == "2023D3":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2023D3XML_cfi')
    
    elif score == "2023D4":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2023D4XML_cfi')

    elif score == "2023D5":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometry2023D5XML_cfi')

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
