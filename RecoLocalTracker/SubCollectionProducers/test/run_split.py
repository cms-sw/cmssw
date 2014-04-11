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
process.MessageLogger.cerr.FwkReport.reportEvery = 10
# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTT_1500_8TeV_Tauola_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_0.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_1.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_2.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_3.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_4.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_5.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_6.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_7.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_8.root',
#    'root://eoscms//eos/cms/store/user/taroni/ClusterSplitting/ZpTauTau8TeV/GEN-SIM-RAW/ZpTauTau_GEN-SIM-RAW_9.root'
#        'file:/afs/cern.ch/work/t/taroni/private/ClusterSplitting/CMSSW_7_1_X_2014-03-18-0200/src/ZpTT_1500_8TeV_Tauola_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'
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
#process.load('RecoLocalTracker.SubCollectionProducers.splitter_tracking_setup2_cff')
from RecoLocalTracker.SubCollectionProducers.splitter_tracking_setup_cff import customizeTracking
customizeTracking('splitClusters', 'splitClusters', 'mySiPixelRecHits', 'mySiStripRecHits')

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
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('ZpTauTau8TeV_split_71X_10.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)
## process.RECOoutput.outputCommands.append( 'keep TrackingParticles_mergedtruth_MergedTrackTruth_*')
## process.RECOoutput.outputCommands.append( 'keep TrackingVertexs_mergedtruth_MergedTrackTruth_*')

# Additional output definition
process.dump = cms.EDAnalyzer("EventContentAnalyzer")
# Other statements

process.GlobalTag.globaltag = 'MC_71_V1::All'



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
