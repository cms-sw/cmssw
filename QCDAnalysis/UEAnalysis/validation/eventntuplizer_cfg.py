
import FWCore.ParameterSet.Config as cms
process = cms.Process("eventntuplizer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
#process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")

#from RecoJets.Configuration.GenJetParticles_cff import *
#from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

## process.chargeParticles = cms.EDFilter("GenParticleSelector",
##     filter = cms.bool(False),
##     src = cms.InputTag("genParticles"),
##     cut = cms.string('charge != 0 & pt > 0.500 & status = 1')
## )

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.threshold = cms.untracked.string('INFO')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source("PoolSource",
    #inputCommands = cms.untracked.vstring("keep *", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap_*_HLT"),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    intputCommands = cms.untracked.vstring("drop *_goodTracks_*_*"),
    fileNames = cms.untracked.vstring(
#run 123596 only reRecoed BSCNOBEAMHALO
     '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7AD8D1B7-2BEA-DE11-B082-00151796D45C.root'
#      'file:/nasdata/yuanchao/cms/QCDAnalysis/Beam/MinimumBias/BeamCommissioning09-Dec9thReReco_SD_AllMinBias-v1_lowpTSkim/5ef010080a33c624ab2cff6acc1de0c6/lowpTRECOSkim_1.root'
#      'file:MinBias_MinBiasPixel_lowpTSkim_1.root'
#BeamHalo
 #   '/store/mc/Summer09/TkBeamHalo/GEN-SIM-RECO/STARTUP31X_V4-v1/0008/66A44C19-4EC5-DE11-BE13-001F29C9E4D0.root',
 # '/store/mc/Summer09/TkBeamHalo/GEN-SIM-RECO/STARTUP31X_V4-v1/0000/78BCFC4A-5EB3-DE11-80EB-001F29C6E900.root'
    )
)

process.TFileService = cms.Service(
    "TFileService",
    ##fileName = cms.string("ntuplizedEvent.root")
    ##fileName = cms.string("ntuplized_ReRecoBSCNOBEAMHALO.root")
    #fileName = cms.string("ntuplized_AllRuns.root")
    fileName = cms.string("ntuplized_090110.root")
 )

selectTracks = cms.EDFilter("RecoTrackSelector",
    src = cms.InputTag("generalTracks"),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(0.18),
    minRapidity = cms.double(-5.0),
    lip = cms.double(0.36),
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(5.0),
    quality = cms.vstring('highPurity'),
    algorithm = cms.vstring(),
    minHit = cms.int32(3),
    min3DHit = cms.int32(0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    vertexCollection = cms.InputTag("offlinePrimaryVertices"),
                            diffvtxbs =cms.double(10.),
    bsuse = cms.bool(False)
)

process.allTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("selectTracks"),
    particleType = cms.string('pi+')
)

process.goodTracks = cms.EDFilter("PtMinCandViewSelector",
    src = cms.InputTag("allTracks"),
    ptMin = cms.double(0.29)
)


process.MyAnalyzer = cms.EDAnalyzer("Event_Ntuplizer",
    particleCollection = cms.InputTag("generalTracks"),
    vertexCollection   = cms.InputTag("offlinePrimaryVertices"),
#    vertexCollection   = cms.InputTag("pixelVertices","","RECO"),
    pixelClusterInput=cms.InputTag("siPixelClusters"),
 
    CaloJetCollectionName = cms.InputTag("sisCone5CaloJets"),
    TrackJetCollectionName = cms.InputTag("ueSisCone5TracksJet"),

      ##particleCollection = cms.InputTag("TrackRefitter1"),
      ##vertexCollection   = cms.InputTag("offlinePrimaryVertices","","Refitting"),
      ##vertexCollection   = cms.InputTag("offlinePrimaryVertices","","REVERTEX"),
      #particleCollection = cms.InputTag("generalTracks","","EXPRESS"),
      #vertexCollection   = cms.InputTag("offlinePrimaryVertices","","EXPRESS"),
      #particleCollection = cms.InputTag("generalTracks","","TOBONLY"),
      #vertexCollection   = cms.InputTag("offlinePrimaryVertices","","TOBONLY"),
    
    #BEAMHALO
    #particleCollection = cms.InputTag("ctfWithMaterialTracksBeamHaloMuon","","RECO"),
    #vertexCollection   = cms.InputTag("","","RECO"),

    #CaloJetCollectionName = cms.InputTag("sisCone5CaloJets"),
    #L1TechPaths_byBit = cms.vint32(40,41),
    #L1TechComb_byBit = cms.string("OR"),#must be -> "OR","AND"
    beamSpot = cms.InputTag("offlineBeamSpot"),
    genEventScale = cms.InputTag("generator"),
    OnlyRECO = cms.bool(True)                               

)

#L1
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

process.L1T1=process.hltLevel1GTSeed.clone()
process.L1T1.L1TechTriggerSeeding = cms.bool(True)
process.L1T1.L1SeedsLogicalExpression = cms.string('(0) AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)')

#HLT
process.load('HLTrigger/HLTfilters/hltHighLevel_cfi')

#MinBiasPixel
process.hltMinBiasPixel1 = process.hltHighLevel.clone()
process.hltMinBiasPixel1.HLTPaths = cms.vstring('HLT_MinBiasPixel_SingleTrack')


process.ueSisCone5TracksJet.jetPtMin = cms.double(0.9)
process.ueSisCone5TracksJet500.jetPtMin = cms.double(0.5)
process.ueSisCone5TracksJet700.jetPtMin = cms.double(0.7)
process.ueSisCone5TracksJet1500.jetPtMin = cms.double(1.5)

#process.p = cms.Path(process.L1T1*process.hltMinBiasPixel1*process.MyAnalyzer)
process.p = cms.Path(process.L1T1*process.hltMinBiasPixel1*((process.selectTracks+process.allTracks+process.goodTracks)*process.UEAnalysisJetsOnlyReco+process.MyAnalyzer))
