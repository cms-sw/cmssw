from __future__ import print_function
import FWCore.ParameterSet.Config as cms

process = cms.Process("TBDtest")

# Messages
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *

#Muon
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "92X_dataRun2_Prompt_v2", '')



########### DATA FILES  ####################################
import FWCore.PythonUtilities.LumiList as LumiList
## Some example files to test ##
lumiSecs = cms.untracked.VLuminosityBlockRange()
#goodLumiSecs = LumiList.LumiList(filename = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/DCSOnly/json_DCSONLY.txt').getCMSSWString().split(',')
#lumiSecs.extend(goodLumiSecs)
readFiles = cms.untracked.vstring()
readFiles.extend(
   [
   '/store/data/Run2017A/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/296/074/00000/22482926-304B-E711-802D-02163E011E07.root',
   '/store/data/Run2017A/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/296/116/00000/8E618859-214C-E711-80E5-02163E01A2CA.root',
   '/store/data/Run2017A/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v2/000/296/168/00000/6CD539ED-664C-E711-8ADB-02163E019B32.root',
   '/store/data/Run2017A/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v2/000/296/172/00000/4848F119-574C-E711-BECE-02163E01A6C9.root',
   '/store/data/Run2017A/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v2/000/296/173/00000/1C60AD2B-6B4C-E711-B956-02163E019D21.root',
   '/store/data/Run2017A/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v2/000/296/174/00000/B4CAA428-6F4C-E711-BE3A-02163E019CC4.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/292/00000/D6E5FC1A-2B59-E711-9CDD-02163E0120DD.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/293/00000/CCBA5DE5-2F59-E711-BE7E-02163E0143BC.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/296/00000/F8F2DBC5-2159-E711-94A0-02163E0127C7.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/308/00000/A4CA2410-1A59-E711-9FCC-02163E0129A2.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/316/00000/9CAAA3AC-0859-E711-9B38-02163E014310.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/359/00000/6EC237E0-8859-E711-A1A4-02163E011939.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/411/00000/64D101B3-185A-E711-98BF-02163E01375A.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/424/00000/C6F85109-525A-E711-878D-02163E014135.root',
   '/store/data/Run2017B/DoubleMuon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/297/425/00000/D0C30579-4F5A-E711-A637-02163E01445D.root'
   ]
)
process.source = cms.Source("PoolSource",
                    lumisToProcess = lumiSecs,
                    fileNames = readFiles
                    )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(15500))
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source.skipEvents=cms.untracked.uint32(0*-1/1)
process.options = cms.untracked.PSet(
  SkipEvent = cms.untracked.vstring( "ProductNotFound" )
)

strflag="ZMuMu"
strflaglower=strflag.lower()
#strflagopts="TBDconstraint:momconstr"
strflagopts="TBDconstraint:fullconstr"
#strflagopts="NOOPTS"

from Alignment.HIPAlignmentAlgorithm.OptionParser.HipPyOptionParser import HipPyOptionParser
from Alignment.HIPAlignmentAlgorithm.OptionParser.HipPyOptionParser import matchPSetsByRecord
from Alignment.HIPAlignmentAlgorithm.OptionParser.HipPyOptionParser import mergeVPSets
optpy=HipPyOptionParser(strflag,strflagopts)

# Track collection name is now interpreted from the flag and may be replaced if an option is specified
# optpy.trkcoll is guaranteed to exist, else python should have already given an error.
strtrackcollname=optpy.trkcoll

# Replace GT specifics if a custom option is specified
if hasattr(optpy, "GlobalTag"):
   process.GlobalTag.globaltag = optpy.GlobalTag
if hasattr(optpy, "GTtoGet"):
   process.GlobalTag.toGet = mergeVPSets(process.GlobalTag.toGet, optpy.GTtoGet, matchPSetsByRecord)

###############################################################################
# Setup common options
strTTRHBuilder = "WithAngleAndTemplate"
if "generic" in optpy.CPEtype: # CPE type is defaulted to "template" in HipPyOptionParser
  strTTRHBuilder = "WithTrackAngle"
###############################################################################

import Alignment.CommonAlignment.tools.trackselectionRefitting as TrackRefitterSequencer

strTBDConstrainer=None
if hasattr(optpy, "TBDconstraint"):
   strtbdconstr=optpy.TBDconstraint
   if "momconstr" in strtbdconstr:
      process.load("RecoTracker.TrackProducer.TwoBodyDecayMomConstraintProducer_cff")
      process.TwoBodyDecayMomConstraint.src = "AlignmentTrackSelector"
      process.TwoBodyDecayMomConstraint.primaryMass = cms.double(91.1876)
      process.TwoBodyDecayMomConstraint.primaryWidth = cms.double(1.4)
      #process.TwoBodyDecayMomConstraint.primaryWidth = cms.double(2.4952)
      #process.TwoBodyDecayMomConstraint.sigmaPositionCut = cms.double(0.07)
      process.TwoBodyDecayMomConstraint.rescaleError = cms.double(1.0)
      process.TwoBodyDecayMomConstraint.chi2Cut = cms.double(99999.)
      #process.TwoBodyDecayMomConstraint.EstimatorParameters.RobustificationConstant = cms.double(1.0)
      strTBDConstrainer="TwoBodyDecayMomConstraint,momentum"

   elif "fullconstr" in strtbdconstr:
      process.load("RecoTracker.TrackProducer.TwoBodyDecayConstraintProducer_cff")
      process.TwoBodyDecayConstraint.src = "AlignmentTrackSelector"
      process.TwoBodyDecayConstraint.primaryMass = cms.double(91.1876)
      process.TwoBodyDecayConstraint.primaryWidth = cms.double(1.4)
      #process.TwoBodyDecayConstraint.primaryWidth = cms.double(2.4952)
      process.TwoBodyDecayConstraint.sigmaPositionCut = cms.double(0.5)
      process.TwoBodyDecayConstraint.rescaleError = cms.double(1.0)
      process.TwoBodyDecayConstraint.chi2Cut = cms.double(99999.)
      #process.TwoBodyDecayConstraint.EstimatorParameters.RobustificationConstant = cms.double(1.0)
      strTBDConstrainer="TwoBodyDecayConstraint,trackParameters"

if strTBDConstrainer is not None:
   print("strTBDConstrainer=",strTBDConstrainer)

process.TrackRefitterSequence = TrackRefitterSequencer.getSequence(
   process,
   strtrackcollname,
   TTRHBuilder = strTTRHBuilder,
   usePixelQualityFlag = None, # Keep default behavior ("WithAngleAndTemplate" -> True, "WithTrackAngle" -> False)
   openMassWindow = False,
   cosmicsDecoMode = False,
   cosmicsZeroTesla = False,
   #momentumConstraint = None, # Should be a momentum constraint object name or None
   momentumConstraint = strTBDConstrainer, # Should be a momentum constraint object name or None
   cosmicTrackSplitting = False,
   use_d0cut = True
   )

# Override TrackRefitterSequencer defaults
process.HighPurityTrackSelector.pMin   = 0.0
process.AlignmentTrackSelector.pMin    = 0.0
process.AlignmentTrackSelector.ptMin   = 15.0
process.AlignmentTrackSelector.etaMin  = -3.0
process.AlignmentTrackSelector.etaMax  = 3.0
process.AlignmentTrackSelector.nHitMin  = 15
process.AlignmentTrackSelector.minHitsPerSubDet.inPIXEL = cms.int32(1)
process.AlignmentTrackSelector.TwoBodyDecaySelector.daughterMass = 0.105658
process.AlignmentTrackSelector.TwoBodyDecaySelector.minXMass = 80.0
process.AlignmentTrackSelector.TwoBodyDecaySelector.maxXMass = 100.0


print(process.TrackRefitterSequence)
subproc=[
   "offlineBeamSpot",
   "HighPurityTrackSelector",
   "FirstTrackRefitter",
   "TrackerTrackHitFilter",
   "HitFilteredTracksTrackFitter",
   "AlignmentTrackSelector",
   "TwoBodyDecayConstraint",
   "TwoBodyDecayMomConstraint",
   "FinalTrackRefitter"
   ]
moduleSum=None
for sp in subproc:
   if hasattr(process, sp):
      print("\n\tAttributes for process.{}".format(sp))
      if moduleSum is None:
         moduleSum=getattr(process,sp)
      else:
         moduleSum+=getattr(process,sp)
      for v in vars(getattr(process,sp)):
         print(v,":",getattr(getattr(process,sp),v))

process.TrackRefitterSequence = cms.Sequence(moduleSum)
print("Final process path:",process.TrackRefitterSequence)
process.p = cms.Path(process.TrackRefitterSequence)


TAG = strflag
if strflagopts:
   TAG = TAG + "_" + strflagopts
TAG = TAG.replace(':','_')
TAG = TAG.strip()
print("Output file:","analyzed_{0}.root".format(TAG))
process.Analyzer = cms.EDAnalyzer(
   "HIPTwoBodyDecayAnalyzer",
   alcarecotracks = cms.InputTag(strtrackcollname),
   refit1tracks = cms.InputTag("FirstTrackRefitter"),
   refit2tracks = cms.InputTag("HitFilteredTracksTrackFitter"),
   finaltracks = cms.InputTag("FinalTrackRefitter")
)

process.options = cms.untracked.PSet(
   SkipEvent = cms.untracked.vstring( "ProductNotFound" )
)
process.TFileService = cms.Service('TFileService',
   fileName=cms.string("analyzed_{0}.root".format(TAG))
)
process.outpath = cms.EndPath(process.Analyzer)
