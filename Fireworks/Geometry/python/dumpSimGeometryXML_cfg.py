import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 2015 2019 PhaseIPixel Phase1_R34F16 Phase2Tk 2023 2023Pixel 2023TTI 2023Muon 2023Muon4Eta 2023HGCal 2023HGCalMuon 2023HGCalMuon4Eta 2023SHCal 2023SHCal4Eta 2023SHCalNoTaper 2023SHCalNoTaper4Eta 2023CFCal 2023CFCal4Eta Phase2TkBE5DPixel10D SLHC DB SLHCDB")
    
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
       process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

    elif score == "DB":
       process.load("Configuration.StandardSequences.GeometryDB_cff")
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']

    else:
       print "Geometry configuration is Geometry.CMSCommonData.cmsExtendedGeometry" + str(score) + "XML_cfi\n"
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry" + str(score) + "XML_cfi")


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
