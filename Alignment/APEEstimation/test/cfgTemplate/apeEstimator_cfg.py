from __future__ import print_function
import os

import FWCore.ParameterSet.Config as cms




##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('sample', 'data1', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Input sample")
options.register('globalTag', "None", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Custom global tag")
options.register('measurementName', "workingArea", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Folder in which to store results")
options.register('fileNumber', 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Input file number")
options.register('iterNumber', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Iteration number")
options.register('lastIter', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Last iteration")
options.register('alignRcd','', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "AlignmentRcd")
options.register('conditions',"None", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "File with conditions")

# get and parse the command line arguments
options.parseArguments()   

print("Input sample: ", options.sample)
print("Input file number", options.fileNumber)
print("Iteration number: ", options.iterNumber)
print("Last iteration: ", options.lastIter)
print("AlignmentRcd: ", options.alignRcd)



##
## Process definition
##
process = cms.Process("ApeEstimator")


process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
from CondCore.CondDB.CondDB_cfi import *

##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.categories.append('SectorBuilder')
process.MessageLogger.categories.append('ResidualErrorBinning')
process.MessageLogger.categories.append('HitSelector')
process.MessageLogger.categories.append('CalculateAPE')
process.MessageLogger.categories.append('ApeEstimator')
process.MessageLogger.categories.append('TrackRefitter')
process.MessageLogger.categories.append('AlignmentTrackSelector')
process.MessageLogger.cerr.threshold = 'WARNING'
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = -1
process.MessageLogger.cerr.SectorBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.HitSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.CalculateAPE = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.ApeEstimator = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.AlignmentTrackSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 ## really show only every 1000th


##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)



##
## Input sample definition
##
isData1 = isData2 = False
isData = False
isQcd = isWlnu = isZmumu = isZtautau = isZmumu10 = isZmumu20 = isZmumu50 = False
isMc = False
isParticleGunMuon = isParticleGunPion = False
isParticleGun = False
if options.sample == 'data1':
    isData = True
elif options.sample == 'data2':
    isData = True
elif options.sample == 'data3':
    isData = True
elif options.sample == 'data4':
    isData = True
elif options.sample == 'qcd':
    isMc = True
elif options.sample == 'wlnu':
    isMc = True
elif options.sample == 'zmumu':
    isMc = True
elif options.sample == 'ztautau':
    isMc = True
elif options.sample == 'zmumu10':
    isMc = True
elif options.sample == 'zmumu20':
    isMc = True
elif options.sample == 'zmumu50':
    isMc = True
elif "MC" in options.sample:
    isMc = True
    print(options.sample)
else:
    print('ERROR --- incorrect data sammple: ', options.sample)
    exit(8888)


##
## Input Files
##
readFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
    fileNames = readFiles
)
readFiles.extend( [
    'file:reco.root',
] )



##
## Number of Events (should be after input file)
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) ) # maxEvents is included in options by default



##
## Check run and event numbers for Dublicates --- only for real data
##
process.source.duplicateCheckMode = cms.untracked.string("checkEachRealDataFile")
#process.source.duplicateCheckMode = cms.untracked.string("checkAllFilesOpened")   # default value


##
## Whole Refitter Sequence
##
process.load("Alignment.APEEstimation.TrackRefitter_38T_cff")

if options.globalTag != "None":
    process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
elif isParticleGun:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')
elif isMc:
    #~ process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_design', '')
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
elif isData:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

if options.conditions != "None":
    import importlib
    mod = importlib.import_module("Alignment.APEEstimation.conditions.{}".format(options.conditions))
    mod.applyConditions(process)

## Alignment and APE
##
## Choose Alignment (w/o touching APE)
if options.alignRcd=='fromConditions':
    pass # Alignment is read from the conditions file in this case
elif options.alignRcd=='design':
    CondDBAlignment = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
    process.myTrackerAlignment = cms.ESSource("PoolDBESSource",
        CondDBAlignment,
        timetype = cms.string("runnumber"),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('TrackerAlignmentRcd'),
                tag = cms.string('TrackerAlignment_Upgrade2017_design_v3')
                )
            )
        )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")

  
elif options.alignRcd == 'misalTest':
    CondDBAlignment = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
    process.myTrackerAlignment = cms.ESSource("PoolDBESSource",
        CondDBAlignment,
        timetype = cms.string("runnumber"),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('TrackerAlignmentRcd'),
                tag = cms.string('TrackerAlignment_Phase1Realignment_CRUZET_2M'),
            )
        )
    )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")
  
elif options.alignRcd == 'mp2705':
    CondDBAlignment = CondDB.clone(connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp2705/jobData/jobm/alignments_MP.db'))
    process.myTrackerAlignment = cms.ESSource("PoolDBESSource",
        CondDBAlignment,
        timetype = cms.string("runnumber"),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('TrackerAlignmentRcd'),
                tag = cms.string('Alignments'),
            )
        )
    )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")

elif options.alignRcd == 'mp2853':
    CondDBAlignment = CondDB.clone()
    process.myTrackerAlignment = cms.ESSource("PoolDBESSource",
        CondDBAlignment,
        timetype = cms.string("runnumber"),
        toGet = cms.VPSet(
            cms.PSet(
                connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp2853/jobData/jobm3/alignments_MP.db'),
                record = cms.string('TrackerAlignmentRcd'),
                tag = cms.string('Alignments'),
            ),
            #~ cms.PSet(
                #~ connect=cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                #~ record=cms.string('SiPixelTemplateDBObjectRcd'),
                #~ tag=cms.string('SiPixelTemplateDBObject_38T_TempForAlignmentReReco2018_v3'),
            #~ )
        )
    )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")
   
elif options.alignRcd == 'hp1370':
    CondDBAlignment = CondDB.clone(connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HIP/xiaomeng/CMSSW_7_4_6_patch5/src/Alignment/HIPAlignmentAlgorithm/hp1370/alignments.db'))
    process.myTrackerAlignment = cms.ESSource("PoolDBESSource",
        CondDBAlignment,
        timetype = cms.string("runnumber"),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('TrackerAlignmentRcd'),
                tag = cms.string('Alignments'),
            )
        )
    )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")
  


elif options.alignRcd == 'globalTag':
  pass
elif options.alignRcd == 'useStartGlobalTagForAllConditions':
  pass
elif options.alignRcd == '':
  pass
else:
  print('ERROR --- incorrect alignment: ', options.alignRcd)
  exit(8888)

## APE
if options.iterNumber!=0:
    CondDBAlignmentError = CondDB.clone(connect = cms.string('sqlite_file:'+os.environ['CMSSW_BASE']+'/src/Alignment/APEEstimation/hists/'+options.measurementName+'/apeObjects/apeIter'+str(options.iterNumber-1)+'.db'))
    process.myTrackerAlignmentErr = cms.ESSource("PoolDBESSource",
        CondDBAlignmentError,
        timetype = cms.string("runnumber"),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                tag = cms.string('APEs')
            )
        )
    )
    process.es_prefer_trackerAlignmentErr = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentErr")
    


##
## Beamspot (Use correct Beamspot for simulated Vertex smearing of ParticleGun)
##
if isParticleGun:
    process.load("Alignment.APEEstimation.BeamspotForParticleGun_cff")


##
## Trigger Selection
##
process.load("Alignment.APEEstimation.TriggerSelection_cff")


##
## ApeEstimator
##
from Alignment.APEEstimation.ApeEstimator_cff import *
process.ApeEstimator1 = ApeEstimator.clone(
    tjTkAssociationMapTag = "TrackRefitterForApeEstimator",
    applyTrackCuts = False,
    analyzerMode = False,
    calculateApe = True,
    Sectors = RecentSectors,
)

process.ApeEstimator2 = process.ApeEstimator1.clone(
  Sectors = ValidationSectors,
  analyzerMode = True,
  calculateApe = False,
)
process.ApeEstimator3 = process.ApeEstimator2.clone(
    zoomHists = False,
)

process.ApeEstimatorSequence = cms.Sequence(process.ApeEstimator1)
if options.iterNumber==0:
  process.ApeEstimatorSequence *= process.ApeEstimator2
  process.ApeEstimatorSequence *= process.ApeEstimator3
elif options.lastIter == True:
  process.ApeEstimatorSequence *= process.ApeEstimator2



##
## Output File Configuration
##
process.TFileService = cms.Service("TFileService",
    fileName = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/'+options.measurementName+'/'+options.sample+str(options.fileNumber)+'.root'),
    closeFileFast = cms.untracked.bool(True)
)



##
## Path
##
process.p = cms.Path(
    #process.TriggerSelectionSequence* # You want to use this if you want to select for triggers
    process.RefitterHighPuritySequence*
    process.ApeEstimatorSequence
)



