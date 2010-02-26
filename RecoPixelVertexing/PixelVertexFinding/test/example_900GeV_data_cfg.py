import FWCore.ParameterSet.Config as cms

process = cms.Process("pvtxana")
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

# Refitters
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.156 $'),
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#======================================
# Global Tag
#======================================
process.GlobalTag.globaltag = 'GR09_R_34X_V5::All'

#======================================
# Input
#======================================
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/E0599DBB-E415-DF11-A592-00304867915A.root'
    ) 
)
process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(-1) )

#===============================================
#output
#===============================================
process.FEVT = cms.OutputModule("PoolOutputModule",
                                    splitLevel = cms.untracked.int32(0),
                                    outputCommands = process.RECOEventContent.outputCommands,
                                    fileName = cms.untracked.string('rerecoOutput.root'),
                                    dataset = cms.untracked.PSet(
            dataTier = cms.untracked.string('RECO'),
                    filterName = cms.untracked.string('')
                )
                                )
#======================================
# Trigger Filter
#======================================
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
process.bit40 = hltLevel1GTSeed.clone(
        L1TechTriggerSeeding = cms.bool(True),
            L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)')
            )

#======================================
# Services
#======================================
process.Timing = cms.Service("Timing")

#======================================
# Analyzer
#======================================

process.test = cms.EDAnalyzer("Analyzer",
        VxInputTag = cms.InputTag("pixelVertices"),  
        AnalysisType = cms.string("Vertex"),
        PrintMsg = cms.bool(False),    
#        IsTrkPart = cms.bool(True),    
        MaxChi2 = cms.double(100),     
        MinPt = cms.double(1.0),       
        VxBound = cms.double(4),      
        VyBound = cms.double(4),     
        VzBound = cms.double(30),     
        MinVxTrkMatch = cms.int32(2),
        MaxTkTracks = cms.int32(2)
        TrackAssociatorByHitsPSet = cms.ParameterSet("TrackAssociatorByHitsPSet")                      
)

process.TFileService = cms.Service("TFileService", 
                           fileName = cms.string("Feb24_vtx_4pm.root"),
                           closeFileFast = cms.untracked.bool(True)
                           )


process.p = cms.Path(process.siPixelRecHits * process.pixelTracks * process.pixelVertices * process.test)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) 
