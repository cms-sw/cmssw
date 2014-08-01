import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 2015 2019 PhaseIPixel SLHCDB SLHC")

def help():
   print "Usage: cmsRun dumpFWRecoGeometry_cfg.py  tag=TAG "
   print "   tag=tagname"
   print "       indentify geometry condition database tag"
   print "      ", varType.keys()
   print ""
   print "   format=formatname"
   print "       dump in plain TTree or in TGeo format, default is TTree"
   print ""
   print "   load=filename"
   print "       a single load instruction, this option excludes 'tag' option"
   print "       for example:" 
   print "       cmsRun dumpFWRecoGeometry_cfg.py load=Configuration.Geometry.GeometryExtended2015Reco_cff" 
   print ""
   exit(1);

def recoGeoLoad(score):
    print "Loading configuration for tag ", options.tag ,"...\n"
    if score == "Run1":
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']
       process.load("Configuration.StandardSequences.GeometryDB_cff")
       process.load("Configuration.StandardSequences.Reconstruction_cff")

    elif score == "2015":
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']
       process.load("Configuration.Geometry.GeometryExtended2015Reco_cff");

    elif  score == "2019":
      from Configuration.AlCa.GlobalTag import GlobalTag
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
      process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')

    elif score ==  "PhaseIPixel":
      from Configuration.AlCa.GlobalTag import GlobalTag
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')
      process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')

    elif score == varType.valueForKey(varType.SLHCDB): # orig dumpFWRecoSLHCGeometry_cfg.py
      process.GlobalTag.globaltag = 'DESIGN42_V17::All'
      process.load("Configuration.StandardSequences.GeometryDB_cff")
      process.load("Configuration.StandardSequences.Reconstruction_cff")

      process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_428SLHCYV0_Phase1_R30F12_HCal_Ideal_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_42X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string('IdealGeometryRecord'),
                 tag = cms.string('TKRECO_Geometry_428SLHCYV0'),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_42X_GEOMETRY")),
        cms.PSet(record = cms.string('PGeometricDetExtraRcd'),
                 tag = cms.string('TKExtra_Geometry_428SLHCYV0'),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_42X_GEOMETRY")),
                 )

    elif score == varType.valueForKey(varType.SLHC): # orig dumpFWRecoGeometrySLHC_cfg.py
      from Configuration.AlCa.autoCond import autoCond
      process.GlobalTag.globaltag = autoCond['mc']
      process.load("Configuration.Geometry.GeometrySLHCSimIdeal_cff")
      process.load("Configuration.Geometry.GeometrySLHCReco_cff")
      process.load("Configuration.StandardSequences.Reconstruction_cff")
      process.trackerSLHCGeometry.applyAlignment = False

    else:
      help()




options = VarParsing.VarParsing ()

options.register ('tag',
                  "2015", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "info about geometry database conditions")


options.register ('format',
                  "TTree", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "write a idToGeo map or make TGeo geometry")


options.register ('load',
                  "", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff")


options.parseArguments()



process = cms.Process("DUMP")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

if not options.load:
   recoGeoLoad(options.tag)
else:
   from Configuration.AlCa.autoCond import autoCond
   process.GlobalTag.globaltag = autoCond['mc']
   process.load(options.load)


tagInfoq = cms.string(options.tag);

if ( options.format == "TGeo"):
    process.add_(cms.ESProducer("FWTGeoRecoGeometryESProducer"))
    process.dump = cms.EDAnalyzer("DumpFWTGeoRecoGeometry")
else:
    process.add_(cms.ESProducer("FWRecoGeometryESProducer"))
    process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                              level   = cms.untracked.int32(1),
                              tagInfo = cms.untracked.string(options.tag)
                              )

process.p = cms.Path(process.dump)
