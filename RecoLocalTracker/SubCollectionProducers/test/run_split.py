import FWCore.ParameterSet.Config as cms

process = cms.Process('SPLIT')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
#process.MessageLogger.cerr.FwkReport.reportEvery = 100
# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
    'root://eoscms///eos/cms/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/02ED271A-DB4A-E311-A756-002618FDA279.root',
    'root://eoscms///eos/cms/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/04FF79F5-DA4A-E311-B585-003048678FDE.root',
    'root://eoscms///eos/cms/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/0EEC6BF3-DA4A-E311-BAAA-0030486792B4.root',
    'root://eoscms///eos/cms/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/18463930-DB4A-E311-83B5-0025905A612A.root',
    'root://eoscms///eos/cms/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/1CDB1236-DB4A-E311-B5DA-002618FDA28E.root',
    'root://eoscms///eos/cms/store/relval/CMSSW_7_0_0_pre8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/20F548F6-DA4A-E311-8B37-00261894397E.root',
),

)
                            
# from Jean-Roch
process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTemplate_cfi")
process.StripCPEfromTrackAngleESProducer = process.StripCPEfromTemplateESProducer.clone(ComponentName='StripCPEfromTrackAngle')

process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTemplate_cfi")
process.StripCPEfromTemplateESProducer.UseTemplateReco = True

# Include the latest pixel templates, which are not in DB. 
# These are to be used for pixel splitting.
process.load('RecoLocalTracker.SiPixelRecHits.PixelCPETemplateReco_cfi')
process.templates.LoadTemplatesFromDB = False

# This is the default speed. Recommended.
process.StripCPEfromTrackAngleESProducer.TemplateRecoSpeed = 0;

# Split clusters have larger errors. Appropriate errors can be 
# assigned by turning UseStripSplitClusterErrors = True. The strip hit pull  
# distributons improve considerably, but it does not help with b-tagging, 
# so leave it False by default 
process.StripCPEfromTrackAngleESProducer.UseStripSplitClusterErrors = True

# Turn OFF track hit sharing
process.load("TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi")
process.trajectoryCleanerBySharedHits.fractionShared = 0.0
process.trajectoryCleanerBySharedHits.allowSharedFirstHit = False
process.load("RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi")
process.simpleTrackListMerger.ShareFrac = 0.0
process.simpleTrackListMerger.allowFirstHitShare = False

# The second step is to split merged clusters.
process.splitClusters = cms.EDProducer(
    "TrackClusterSplitter",
    stripClusters         = cms.InputTag("siStripClusters::SPLIT"),
    pixelClusters         = cms.InputTag("siPixelClusters::SPLIT"),
    useTrajectories       = cms.bool(False),
    trajTrackAssociations = cms.InputTag('generalTracks::SPLIT'),
    tracks                = cms.InputTag('pixelTracks::SPLIT'),
    propagator            = cms.string('AnalyticalPropagator'),
    vertices              = cms.InputTag('pixelVertices::SPLIT'),
    simSplitPixel         = cms.bool(False), # ideal pixel splitting turned OFF
    simSplitStrip         = cms.bool(False), # ideal strip splitting turned OFF
    tmpSplitPixel         = cms.bool(True), # template pixel spliting
    tmpSplitStrip         = cms.bool(True), # template strip splitting
    useStraightTracks     = cms.bool(True),
    test     = cms.bool(True)
    )

process.mySiPixelRecHits = process.siPixelRecHits.clone(src = cms.InputTag("splitClusters"))
process.mySiStripRecHits = process.siStripMatchedRecHits.clone(
    src = cms.InputTag("splitClusters"),  ClusterProducer = cms.InputTag("splitClusters")
    )

############################## inserted new stuff
                            
# from Jean-Roch 
process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTemplate_cfi")
process.StripCPEfromTrackAngleESProducer = process.StripCPEfromTemplateESProducer.clone(ComponentName='StripCPEfromTrackAngle')

process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTemplate_cfi")
process.StripCPEfromTemplateESProducer.UseTemplateReco = True

# Include the latest pixel templates, which are not in DB. 
# These are to be used for pixel splitting.
process.load('RecoLocalTracker.SiPixelRecHits.PixelCPETemplateReco_cfi')
process.templates.LoadTemplatesFromDB = False

# This is the default speed. Recommended.
process.StripCPEfromTrackAngleESProducer.TemplateRecoSpeed = 0;

############################## inserted new stuff

process.newrechits = cms.Sequence(process.mySiPixelRecHits*process.mySiStripRecHits)

######## track to vertex assoc ##################3
from CommonTools.RecoUtils.pf_pu_assomap_cfi import AssociationMaps
process.Vertex2TracksDefault = AssociationMaps.clone(
    AssociationType = cms.InputTag("VertexToTracks"),
    MaxNumberOfAssociations = cms.int32(1)
)

# The commands included in splitter_tracking_setup_cff.py instruct 
# the tracking machinery to use the clusters and rechits generated after 
# cluster splitting (instead of the default clusters and rechits)
process.load('RecoLocalTracker.SubCollectionProducers.splitter_tracking_setup_cff')

process.fullreco = cms.Sequence(process.globalreco*process.highlevelreco)
process.Globalreco = cms.Sequence(process.globalreco)
process.Highlevelreco = cms.Sequence(process.highlevelreco)

process.options = cms.untracked.PSet(

)

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

# Output definition
process.PixelTreeSplit = cms.EDAnalyzer(
    "PixelAnalysisTree",
    verbose                = cms.untracked.int32(0),
    rootFileName           = cms.untracked.string('pixelTrees_TS_DATA.root'),
    treeName               = cms.untracked.string('treeTmpSplit'),
    dumpAllEvents          = cms.untracked.int32(1),
    globalTag              = cms.string("GR_R_53::All"),
    muonCollectionLabel    = cms.untracked.InputTag('muons'),
#    trajectoryInputLabel   = cms.untracked.InputTag('TrackRefitter'),
    trajectoryInputLabel   = cms.untracked.InputTag('generalTracks::SPLIT'),          
    trackCollectionLabel   = cms.untracked.InputTag('generalTracks::SPLIT'),
    pixelClusterLabel      = cms.untracked.InputTag('splitClusters'),
    pixelRecHitLabel       = cms.untracked.InputTag('mySiPixelRecHits'),
    L1GTReadoutRecordLabel = cms.untracked.InputTag("gtDigis"),
    hltL1GtObjectMap       = cms.untracked.InputTag("hltL1GtObjectMap"),          
    HLTResultsLabel        = cms.untracked.InputTag("TriggerResults::HLT"),
    HLTProcessName         = cms.untracked.string("HLT"),
    mapTrack2Vertex        = cms.untracked.bool(True)
    )


process.RECOoutput = cms.OutputModule("PoolOutputModule",
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('TTbar700_split.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO')
    )
)
## process.RECOoutput.outputCommands.append( 'keep TrackingParticles_mergedtruth_MergedTrackTruth_*')
## process.RECOoutput.outputCommands.append( 'keep TrackingVertexs_mergedtruth_MergedTrackTruth_*')

# Additional output definition
process.dump = cms.EDAnalyzer("EventContentAnalyzer")
# Other statements

process.GlobalTag.globaltag = 'START70_V1::All'
#process.GlobalTag.globaltag = 'GR_R_52_V7::All'


# Path and EndPath definitions
process.init_step = cms.Path(cms.Sequence(process.RawToDigi*process.localreco*process.offlineBeamSpot+process.recopixelvertexing))
process.dump_step = cms.Path(process.dump)
process.splitClusters_step=cms.Path(process.splitClusters)
process.newrechits_step=cms.Path(process.newrechits)
process.fullreco_step=cms.Path(process.fullreco)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)
#process.pixeltree_tempsplit =cms.Path(process.PixelTreeSplit)
#process.vertex_assoc = cms.Path(process.Vertex2TracksDefault)

# Schedule definition

process.schedule = cms.Schedule(process.init_step,process.splitClusters_step,process.newrechits_step,process.fullreco_step,process.RECOoutput_step)
