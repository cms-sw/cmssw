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
    from Configuration.AlCa.autoCond import autoCond

    if score == "Run1":
       process.GlobalTag.globaltag = autoCond['mc']
       process.load("Configuration.StandardSequences.GeometryDB_cff")
       process.load("Configuration.StandardSequences.Reconstruction_cff")

    elif score == "2015":
       process.GlobalTag.globaltag = autoCond['mc']
       process.load("Configuration.Geometry.GeometryExtended2015Reco_cff");

    elif score == "2017":
        process.GlobalTag.globaltag = autoCond['run2_design']
        process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                            tag = cms.string('TKParameters_Geometry_Run2_75YV1'),
                                            connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                            )
                                   )
        process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')

    elif  score == "2019":
      process.GlobalTag.globaltag = autoCond['run2_design']
      process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                            tag = cms.string('TKParameters_Geometry_Run2_75YV1'),
                                            connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                            )
                                   )
      process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')

    elif score ==  "PhaseIPixel":
      process.GlobalTag.globaltag = autoCond['run2_design']
      process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                            tag = cms.string('TKParameters_Geometry_Run2_75YV1'),
                                            connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                            )
                                   )
      process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')

    elif  score == "2023":
      process.GlobalTag.globaltag = autoCond['run2_design']
      process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                            tag = cms.string('TKParameters_Geometry_Run2_75YV1'),
                                            connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                            )
                                   )
      process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
      
    elif  score == "2023Muon":
      process.GlobalTag.globaltag = autoCond['run2_design']
      process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                            tag = cms.string('TKParameters_Geometry_Run2_75YV1'),
                                            connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                            )
                                   )
      process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')

    elif  score == "GEMDev":
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
print "Using GlobalTag =",process.GlobalTag.globaltag
process.p = cms.Path(process.dump)
