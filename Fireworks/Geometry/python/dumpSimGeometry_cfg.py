import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 2015 PhaseIPixel Phase1_R34F16 Phase2Tk SLHCDB SLHC DB")

def help():
   print "Usage: cmsRun dumpSimGeometry_cfg.py  tag=TAG "
   print "   tag=tagname"
   print "       indentify geometry condition database tag"
   print "      ", varType.keys()
   print ""
   print "   load=filename"
   print "       a single load instruction, this option excludes 'tag' option"
   print "       e.g load=Geometry.CMSCommonData.Phase1_R34F16_cmsSimIdealGeometryXML_cff"
   print 
   exit(1);

def simGeoLoad(score):
    print "Loading configuration for tag ", options.tag ,"...\n"
    if score == "Run1":
       process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

    elif score == "2015":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015XML_cfi")

    elif score == "PhaseIPixel":
       process.load('Geometry.CMSCommonData.GeometryExtendedPhaseIPixel_cfi')

    elif score == "Phase2Tk":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometryPhase2TkBEXML_cfi')

    elif score == "SLHCDB":
        process.load("Configuration.StandardSequences.GeometryDB_cff")
        process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
        process.GlobalTag.globaltag = 'DESIGN42_V17::All'
        process.XMLFromDBSource.label=''

        process.GlobalTag.toGet = cms.VPSet(
                 cms.PSet(record = cms.string("GeometryFileRcd"),
                    tag = cms.string("XMLFILE_Geometry_428SLHCYV0_Phase1_R30F12_HCal_Ideal_mc"),
                    connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_42X_GEOMETRY")
                 )
        )

    elif score == "DB":
        process.load("Configuration.StandardSequences.GeometryDB_cff")
        process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
        from Configuration.AlCa.autoCond import autoCond
        process.GlobalTag.globaltag = autoCond['mc']

    elif score == "SLHC":
        process.load('SLHCUpgradeSimulations.Geometry.Phase1_R30F12_HCal_cmsSimIdealGeometryXML_cff')
        

    elif score == "Phase1_R34F16":
        process.load('Geometry.CMSCommonData.Phase1_R34F16_cmsSimIdealGeometryXML_cff')
        
    else:
      help()



options = VarParsing.VarParsing ()

options.register ('tag',
                  "2015", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "info about geometry database conditions")

options.register ('load',
                  "", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "load path e.g Geometry.CMSCommonData.Phase1_R34F16_cmsSimIdealGeometryXML_cff")


options.parseArguments()


process = cms.Process("SIMDUMP")

if not options.load:
   simGeoLoad(options.tag)
else:
   process.load(options.load)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level = cms.untracked.int32(14)
))

process.dump = cms.EDAnalyzer("DumpSimGeometry")

process.p = cms.Path(process.dump)
