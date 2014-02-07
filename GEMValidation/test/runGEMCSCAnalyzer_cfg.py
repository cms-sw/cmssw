import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMCSCANA")

## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## TrackingComponentsRecord required for matchers
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')

## GEM geometry customization
use6part = True
if use6part:
  from Geometry.GEMGeometry.gemGeometryCustoms import custom_GE11_6partitions_v1
  process = custom_GE11_6partitions_v1(process)

## input
from GEMCode.SimMuL1.GEMCSCTriggerSamplesLib import files
suffix = '_gem98_pt2-50_PU0_pt0_new'
#inputDir = files[suffix]
inputDir = ['/pnfs/cms/WAX/11/store/user/tahuang/tahuang/SingleMuPt2-50Fwdv2_1M/SingleMuPt2-50Fwdv2_L1_PU140_Pt0_LCT2_pretrig3_trig3_noLQclcts/e46e45c13b64e2c906f8d4c7d3ce8b26/']
#inputDir = ['/pnfs/cms/WAX/11/store/user/tahuang/tahuang/SingleMuPt2-50Fwdv2_1M/SingleMuPt2-50Fwdv2_L1_PU400_Pt0_LCT2_pretrig3_trig3_noLQclcts/4b06d8f3849ae4eb70f8b3620dc31163/']

theInputFiles = []
import os
for d in range(len(inputDir)):
  my_dir = inputDir[d]
  if not os.path.isdir(my_dir):
    print "ERROR: This is not a valid directory: ", my_dir
    if d==len(inputDir)-1:
      print "ERROR: No input files were selected"
      exit()
    continue
  print "Proceed to next directory"
  ls = os.listdir(my_dir)
  ## this works only if you pass the location on pnfs - FIXME for files staring with store/user/... 
  theInputFiles.extend([my_dir[16:] + x for x in ls if x.endswith('root')])

theInputFiles = ['file:/uscms_data/d3/tahuang/RunCrab/CMSSW_6_2_0_SLHC7/src/GEMCode/SimMuL1/test/out_L1.root'] 
print "InputFiles: ", theInputFiles

##output file name
#theoutputFiles = 'gem_PU140_pretrig3_trig3_noLQclcts_minHits3.root'
theoutputFiles = "out_GEMCSC_Ana.root"
print "OutputFiles: ", theoutputFiles

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

# the analyzer configuration
from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching
process.GEMCSCAnalyzer = cms.EDAnalyzer("GEMCSCAnalyzer",
    verbose = cms.untracked.int32(0),
    stationsToUse = cms.vint32(1,2,3,4),
    simTrackMatching = simTrackMatching
)
matching = process.GEMCSCAnalyzer.simTrackMatching
matching.simTrack.minPt = 1.5
matching.cscSimHit.minNHitsChamber = 3
matching.cscStripDigi.minNHitsChamber = 3
matching.cscWireDigi.minNHitsChamber = 3
matching.cscCLCT.minNHitsChamber = 3
matching.cscALCT.minNHitsChamber = 3
matching.cscLCT.minNHitsChamber = 3
matching.cscMPLCT.minNHitsChamber = 3
matching.gemRecHit.input = ""
matching.tfTrack.input = ""
matching.tfCand.input = ""
matching.gmtCand.input = ""
matching.l1Extra.input = ""

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        *theInputFiles
    )
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(theoutputFiles)
)

process.p = cms.Path(process.GEMCSCAnalyzer)

