import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 2015 2017 2019 PhaseIPixel 2023Muon SLHC SLHCDB")

def help():
   print "Usage: cmsRun dumpFWRecoGeometry_cfg.py  tag=TAG "
   print "   tag=tagname"
   print "       indentify geometry condition database tag"
   print "      ", varType.keys()
   print ""
   print "   tgeo=bool"
   print "       dump in TGeo format to borwse it geomtery viewer"
   print "       import this will in Fireworks with option --sim-geom-file"
   print ""
   print ""
   exit(1);

def recoGeoLoad(score):
    print "Loading configuration for tag ", options.tag ,"...\n"
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

    if score == "Run1":
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']
       process.load("Configuration.StandardSequences.GeometryDB_cff")
       process.load("Configuration.StandardSequences.Reconstruction_cff")

    elif score == "2015":
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']
       process.load("Configuration.Geometry.GeometryExtended2015Reco_cff");

    elif score == "2017":
       from Configuration.AlCa.GlobalTag import GlobalTag
       process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2017', '')
       process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
       # Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
       #from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2017 
       #process.load("SLHCUpgradeSimulations.Configuration.combinedCustoms.cust_2017")
      # process = cust_2017(process)

    elif  score == "2019":
      from Configuration.AlCa.GlobalTag import GlobalTag
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
      process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
      #from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2019

    elif score ==  "PhaseIPixel":
      from Configuration.AlCa.GlobalTag import GlobalTag
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')
      process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')

    elif  score == "2023":
      from Configuration.AlCa.autoCond import autoCond
      process.GlobalTag.globaltag = autoCond['mc']
      #from Configuration.AlCa.GlobalTag import GlobalTag
      #process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V6::All', '')
      process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
      #from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2019
      #process = cust_2019(process)
      
    elif  score == "2023Muon":
      from Configuration.AlCa.autoCond import autoCond
      process.GlobalTag.globaltag = autoCond['mc']
      #from Configuration.AlCa.GlobalTag import GlobalTag
      #process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V6::All', '')
      process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
      # Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
      #from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023Muon
      #call to customisation function cust_2023Muon imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
      #process = cust_2023Muon(process)

    elif  score == "GEMDev":
      from Configuration.AlCa.autoCond import autoCond
      process.GlobalTag.globaltag = autoCond['mc']
      process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')

    elif score == "SLHCDB": # orig dumpFWRecoSLHCGeometry_cfg.py
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

    elif score == varType.valueForKey("SLHC"): # orig dumpFWRecoGeometrySLHC_cfg.py
      from Configuration.AlCa.autoCond import autoCond
      process.GlobalTag.globaltag = autoCond['mc']
      process.load("Configuration.Geometry.GeometrySLHCSimIdeal_cff")
      process.load("Configuration.Geometry.GeometrySLHCReco_cff")
      process.load("Configuration.StandardSequences.Reconstruction_cff")
      process.trackerSLHCGeometry.applyAlignment = False

    else:
      help()




options = VarParsing.VarParsing ()


defaultOutputFileName="cmsRecoGeom.root"

options.register ('tag',
                  "2015", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "tag info about geometry database conditions")


options.register ('tgeo',
                  False, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "write geometry in TGeo format")


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
    process.add_(cms.ESProducer("FWTGeoRecoGeometryESProducer"))
    process.dump = cms.EDAnalyzer("DumpFWTGeoRecoGeometry",
                              tagInfo = cms.untracked.string(options.tag),
                       outputFileName = cms.untracked.string(options.out)
                              )
else:
    if (options.out == defaultOutputFileName ):
        options.out = "cmsRecoGeom-" +  str(options.tag) + ".root"
    process.add_(cms.ESProducer("FWRecoGeometryESProducer"))
    process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                              level   = cms.untracked.int32(1),
                              tagInfo = cms.untracked.string(options.tag),
                       outputFileName = cms.untracked.string(options.out)
                              )

print "Dumping geometry in " , options.out, "\n"; 
process.p = cms.Path(process.dump)
