import FWCore.ParameterSet.Config as cms
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.Utilities.Enumerate import Enumerate

varType = Enumerate ("Run1 2015 2017 2019 PhaseIPixel 2023 2023Muon GEMDev MaPSA SLHC")

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
   print "   tracker=bool"
   print "       include Tracker subdetectors"
   print ""
   print "   muon=bool"
   print "       include Muon subdetectors"
   print ""
   print "   calo=bool"
   print "       include Calo subdetectors"
   print ""
   print ""
   exit(1);

def recoGeoLoad(score):
    print "Loading configuration for tag ", options.tag ,"...\n"

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
       # Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
       #from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2017 
       #process.load("SLHCUpgradeSimulations.Configuration.combinedCustoms.cust_2017")
       #process = cust_2017(process)

    elif  score == "2019":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['upgrade2019']
       ## NOTE: There is no PTrackerParameters Rcd in this GT yet
       process.load('Geometry.TrackerGeometryBuilder.trackerParameters_cfi')
       process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
       ## NOTE: There are no Muon alignement records in the GT yet
       process.DTGeometryESModule.applyAlignment = cms.bool(False)
       process.CSCGeometryESModule.applyAlignment = cms.bool(False)

    elif score == "PhaseIPixel":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.GlobalTag import GlobalTag
       process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')
       process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')

    elif  score == "2023":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']
      #from Configuration.AlCa.GlobalTag import GlobalTag
      #process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V6::All', '')
       process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
      #from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2019
      #process = cust_2019(process)
      
    elif  score == "2023Muon":
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
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
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
       from Configuration.AlCa.autoCond import autoCond
       process.GlobalTag.globaltag = autoCond['mc']
       process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')

    elif score == "MaPSA":
       process.load('Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff')
       process.load('Geometry.TrackerCommonData.mapsaGeometryXML_cfi')
       process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
       process.load('Geometry.TrackerNumberingBuilder.trackerTopology_cfi')
       process.load('Geometry.TrackerGeometryBuilder.trackerParameters_cfi')
       process.load('Geometry.TrackerGeometryBuilder.trackerGeometry_cfi')
       process.trackerGeometry.applyAlignment = cms.bool(False)
       process.load('RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi')

       process.load('Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi')

    elif score == "SLHC": # orig dumpFWRecoGeometrySLHC_cfg.py
       process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
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
                 Calo = cms.untracked.bool(options.calo)))
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
                 Calo = cms.untracked.bool(options.calo)))
    process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                              level   = cms.untracked.int32(1),
                              tagInfo = cms.untracked.string(options.tag),
                       outputFileName = cms.untracked.string(options.out)
                              )

print "Dumping geometry in " , options.out, "\n"; 
process.p = cms.Path(process.dump)
