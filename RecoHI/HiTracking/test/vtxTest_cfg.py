import FWCore.ParameterSet.Config as cms

process = cms.Process('vtxTEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('vtxTest nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'/store/hidata/HIRun2010/HICorePhysics/RECO/PromptReco-v3/000/151/353/94B06736-30F2-DF11-B1EC-003048D37560.root'
    #'file:hiRecoDM_RECO.root'
    #'file:highMultOneSideSkim.root'
    'file:/d01/edwenger/test/CMSSW_3_9_2_patch5/src/mergeDiJet_run151153.root'
    )
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.Timing = cms.Service("Timing")

# Output definition

process.RECOoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('vtxTest_RAW2DIGI_RECO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('testVtx.root'))

# Other statements
process.GlobalTag.globaltag = 'GR10_P_V12::All'

# Vertex Analyzer
process.ClusterVertexAnalyzer = cms.EDAnalyzer("HIPixelClusterVtxAnalyzer",
                                      pixelRecHits=cms.InputTag("siPixelRecHits"),
                                      minZ=cms.double(-30.0),
                                      maxZ=cms.double(30.05),
                                      zStep=cms.double(0.1),
                                      maxHists=cms.int32(10)         
                                      )

# Trigger and Event Selection
process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")
process.hltMinBiasHFOrBSC = process.hltHighLevel.clone()
process.hltMinBiasHFOrBSC.HLTPaths = ["HLT_HIMinBiasHfOrBSC_Core"]
process.load("HeavyIonsAnalysis.Configuration.collisionEventSelection_cff")

process.pixelActivityFilter = cms.EDFilter( "HLTPixelActivityFilter",
   inputTag    = cms.InputTag( "siPixelClusters" ),
   minClusters = cms.uint32( 4000 ),
   maxClusters = cms.uint32( 0 )                                    
)

#process.evtSel = cms.Sequence(process.hltMinBiasHFOrBSC*process.collisionEventSelection)
process.evtSel = cms.Sequence(process.pixelActivityFilter)

# Path and EndPath definitions

process.filter_step = cms.Path(process.evtSel)

process.reconstruction_step = cms.Path(#process.evtSel *
                                       process.siPixelRecHits * process.ClusterVertexAnalyzer)

process.endjob_step = cms.Path(process.evtSel * process.endOfProcess)

process.RECOoutput_step = cms.EndPath(process.evtSel * process.RECOoutput)


# Schedule definition
process.schedule = cms.Schedule(#process.filter_step,
                                process.reconstruction_step)#,process.endjob_step,process.RECOoutput_step)
