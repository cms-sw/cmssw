import os

import FWCore.ParameterSet.Config as cms




##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('sample', 'data1', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Input sample")
options.register('fileNumber', 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Input file number")
options.register('iterNumber', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Iteration number")
options.register('lastIter', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Last iteration")
options.register('alignRcd','', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "AlignmentRcd")



# get and parse the command line arguments
if( hasattr(sys, "argv") ):
    for args in sys.argv :
        arg = args.split(',')
        for val in arg:
            val = val.split('=')
            if(len(val)==2):
                setattr(options,val[0], val[1])

print "Input sample: ", options.sample
print "Input file number", options.fileNumber
print "Iteration number: ", options.iterNumber
print "Last iteration: ", options.lastIter
print "AlignmentRcd: ", options.alignRcd



##
## Process definition
##
process = cms.Process("ApeEstimator")


process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag

# new CPE

from RecoLocalTracker.SiStripRecHitConverter.StripCPEgeometric_cfi import *
TTRHBuilderGeometricAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
StripCPE = cms.string('StripCPEfromTrackAngle'), # cms.string('StripCPEgeometric'),
#StripCPE = cms.string('StripCPEgeometric'),
ComponentName = cms.string('WithGeometricAndTemplate'),
PixelCPE = cms.string('PixelCPEGeneric'),
#PixelCPE = cms.string('PixelCPETemplateReco'),
Matcher = cms.string('StandardMatcher'),
ComputeCoarseLocalPositionFromDisk = cms.bool(False)
)



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
else:
    print 'ERROR --- incorrect data sammple: ', options.sample
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
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )



##
## Check run and event numbers for Dublicates --- only for real data
##
process.source.duplicateCheckMode = cms.untracked.string("checkEachRealDataFile")
#process.source.duplicateCheckMode = cms.untracked.string("checkAllFilesOpened")   # default value



##
## Whole Refitter Sequence
##
process.load("Alignment.APEEstimation.TrackRefitter_38T_cff")

if isParticleGun:
    process.GlobalTag.globaltag = 'DESIGN42_V12::All'
elif isMc:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')
    
    ##### To be used when running on Phys14MC with a CMSSW version > 72X
#    process.GlobalTag.toGet = cms.VPSet(
#		cms.PSet(
#			record = cms.string("BeamSpotObjectsRcd"),
#			tag = cms.string("Realistic8TeVCollisions_START50_V13_v1_mc"),
#			connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
#		)
#	)


elif isData:
    #~ process.GlobalTag.globaltag = 'GR_P_V56'
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')




## Alignment and APE
##
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
## Choose Alignment (w/o touching APE)
if options.alignRcd=='design':
  process.myTrackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'frontier://FrontierProd/CMS_CONDITIONS',
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerAlignment_Ideal62X_mc')
      )
    )
  )
  process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")

  
elif options.alignRcd == 'misalTest':
  process.myTrackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'sqlite_file:/afs/cern.ch/user/c/cschomak/CMSSW_7_4_1/src/Alignment/APEEstimation/test/geometry_MisalignmentScenario_20mu_fromDESRUN2_74_V4.db',
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Alignments'),
      )
    )
  )
  process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")
  
elif options.alignRcd == 'mp1799':
  process.myTrackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1799/jobData/jobm/alignments_MP.db',
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Alignments'),
      )
    )
  )
  process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")
   
elif options.alignRcd == 'hp1370':
  process.myTrackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = 'sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HIP/xiaomeng/CMSSW_7_4_6_patch5/src/Alignment/HIPAlignmentAlgorithm/hp1370/alignments.db',
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
  print 'ERROR --- incorrect alignment: ', options.alignRcd
  exit(8888)

## APE
if options.iterNumber!=0:
    process.myTrackerAlignmentErr = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
      connect = 'sqlite_file:'+os.environ['CMSSW_BASE']+'/src/Alignment/APEEstimation/hists/apeObjects/apeIter'+str(options.iterNumber-1)+'.db',
      toGet = [
        cms.PSet(
          record = cms.string('TrackerAlignmentErrorExtendedRcd'),
          tag = cms.string('TrackerAlignmentExtendedErr_2009_v2_express_IOVs')
        ),
      ],
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
    fileName = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/workingArea/'+options.sample+str(options.fileNumber)+'.root'),
    closeFileFast = cms.untracked.bool(True)
)



##
## Path
##
process.p = cms.Path(
    process.TriggerSelectionSequence*
    process.RefitterHighPuritySequence*
    process.ApeEstimatorSequence
)

