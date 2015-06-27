import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 Ideal2015 Ideal2015dev 2015 2015dev 2019 PhaseIPixel Phase1_R34F16 Phase2Tk  2023Muon SLHC DB SLHCDB")

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

    elif score == "2015":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015XML_cfi")

    elif score == "2015dev":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015devXML_cfi")

    elif score == "Ideal2015":
       process.load("Geometry.CMSCommonData.cmsIdealGeometry2015XML_cfi")

    elif score == "Ideal2015dev":
       process.load("Geometry.CMSCommonData.cmsIdealGeometry2015devXML_cfi")

    elif score == "RPC4RE11":
       process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015XML_RPC4RE11_cfi")

    elif score == "2017":
       process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
       
    elif score == "2019":
       process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
  
    elif score == "PhaseIPixel":
       process.load('Geometry.CMSCommonData.GeometryExtendedPhaseIPixel_cfi')

    elif score == "Phase1_R34F16":
        process.load('Geometry.CMSCommonData.Phase1_R34F16_cmsSimIdealGeometryXML_cff')
 
    elif score == "Phase2Tk":
       process.load('Geometry.CMSCommonData.cmsExtendedGeometryPhase2TkBEXML_cfi')

    elif score == "2023Muon":
       process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')

    elif score == "2023":
       process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
        
    elif score == "SLHC":
        process.load('SLHCUpgradeSimulations.Geometry.Phase1_R30F12_HCal_cmsSimIdealGeometryXML_cff')
        
    elif score == "DB":
        process.load("Configuration.StandardSequences.GeometryDB_cff")
        process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
        from Configuration.AlCa.autoCond import autoCond
        process.GlobalTag.globaltag = autoCond['mc']

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
