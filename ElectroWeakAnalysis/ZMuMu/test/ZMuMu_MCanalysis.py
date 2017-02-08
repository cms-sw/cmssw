import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuMCanalysis")
#process.load("ElectroWeakAnalysis.Skimming.mcTruthForDimuons_cff")
#process.load("ElectroWeakAnalysis/Skimming/zMuMu_SubskimPathsWithMCTruth_cff")
#adapting what we have in zMuMu_SubskimPathsWithMCTruth_cff, we don't need to add user data....


process.load("ElectroWeakAnalysis.Skimming.patCandidatesForZMuMuSubskim_cff")
#### quality cut to apply or not
process.patAODTrackCands.cut = 'pt > 20.'
process.selectedPatTracks.cut = 'pt > 20. & abs(eta)<2.1 & ( track().hitPattern().numberOfValidStripHits  + track().hitPattern().numberOfValidPixelHits)>= 10 & track().normalizedChi2()<10.'

process.selectedPatMuons.cut = 'pt > 20. & abs(eta)<2.1 & ( (isGlobalMuon & (globalTrack().hitPattern().numberOfValidStripHits  + globalTrack().hitPattern().numberOfValidPixelHits)>= 10 & globalTrack().normalizedChi2()<10. & outerTrack().hitPattern().numberOfValidMuonHits>0 ) ||  (isStandAloneMuon & outerTrack().normalizedChi2()<10. & outerTrack().hitPattern().numberOfValidMuonHits>0 ))'


### temporarly form 31X-->35X reprocessed spring10 data
process.patTrigger.processName = "REDIGI"
process.patTriggerEvent.processName = "REDIGI"
process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI" )
process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI" )



process.load("ElectroWeakAnalysis.Skimming.dimuons_cfi")
process.load("ElectroWeakAnalysis.Skimming.dimuonsOneTrack_cfi")
process.load("ElectroWeakAnalysis.Skimming.dimuonsGlobal_cfi")
process.load("ElectroWeakAnalysis.Skimming.dimuonsOneStandAloneMuon_cfi")

# MC matching sequence
process.load("ElectroWeakAnalysis.Skimming.mcTruthForDimuons_cff")
process.goodMuonMCMatch.src = 'selectedPatMuonsTriggerMatch'
process.goodTrackMCMatch.src = 'selectedPatTracks'






process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1000




process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('START37_V1A::All')
process.GlobalTag.globaltag = cms.string('MC_3XY_V26::All')
process.load("Configuration.StandardSequences.MagneticField_cff")



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#'file:/tmp/degrutto/testDimuonSkim_all.root'
#'rfio:/castor/cern.ch/user/f/fabozzi/testsubskimMC/testZMuMuSubskim.root'
'rfio:/castor/cern.ch/user/f/fabozzi/mc7tev/spring10/38262142-DF46-DF11-8238-0030487C6A90.root'

#'rfio:/castor/cern.ch/user/f/fabozzi/mc7tev/F8EE38AF-1EBE-DE11-8D19-00304891F14E.root'

    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('zMuMu_MCanalysis.root')
)

process.zToMuMu = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuons"),
    cut = cms.string('daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1 & charge=0'),
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    overlap = cms.InputTag("zToMuMu"),
)

process.zToMuMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("dimuonsOneTrack"),
    cut = cms.string('daughter(0).isGlobalMuon = 1 & charge=0'),
)

process.goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("zToMuMu"),
)


process.zMuMu_MCanalyzer = cms.EDFilter("ZMuMu_MCanalyzer",
    muons = cms.InputTag("selectedPatMuons"),
    tracks = cms.InputTag("selectedPatTracks"),
    zMuMu = cms.InputTag("zToMuMu"),
    zMuStandAlone = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    zMuTrack = cms.InputTag("goodZToMuMuOneTrack"),
    zMuMuMatchMap = cms.InputTag("allDimuonsMCMatch"),
    zMuStandAloneMatchMap = cms.InputTag("allDimuonsMCMatch"),
    zMuTrackMatchMap = cms.InputTag("dimuonsOneTrackMCMatch"),
    genParticles = cms.InputTag("genParticles"),
    bothMuons = cms.bool(True),                              
    zMassMin = cms.untracked.double(60.0),
    zMassMax = cms.untracked.double(120.0),
    etamin = cms.untracked.double(0.0),                                        
    etamax = cms.untracked.double(2.1),
    ptmin = cms.untracked.double(20.0),
    hltPath = cms.untracked.string("HLT_Mu9"),
 ###isolation block
    isomax = cms.untracked.double(3.0),
    ptThreshold = cms.untracked.double(1.5),
    etEcalThreshold = cms.untracked.double(0.2),
    etHcalThreshold = cms.untracked.double(0.5),
    deltaRVetoTrk = cms.untracked.double(0.015),
    deltaRTrk = cms.untracked.double(0.3),
    deltaREcal = cms.untracked.double(0.25),
    deltaRHcal = cms.untracked.double(0.25),
    alpha = cms.untracked.double(0.),
    beta = cms.untracked.double(-0.75),
    relativeIsolation=cms.untracked.bool(False)
)


process.eventInfo = cms.OutputModule("AsciiOutputModule")


process.dimuonsPath = cms.Path(
                       process.goodMuonRecoForDimuon *
                       process.dimuons *
                       process.mcTruthForDimuons *
                       process.dimuonsGlobal *
                       process.dimuonsOneStandAloneMuon 
                       )

process.dimuonsOneTrackPath = cms.Path(
                               process.goodMuonRecoForDimuon*
                               process.dimuonsOneTrack*
                               process.mcTruthForDimuonsOneTrack 
                               )




process.p = cms.Path(#process.mcTruthForDimuons *
                     process.zToMuMu *
                     process.goodZToMuMuOneStandAloneMuon *                     
                     process.zToMuMuOneTrack *
                     process.goodZToMuMuOneTrack *                     
                     process.zMuMu_MCanalyzer)
process.e = cms.EndPath(process.eventInfo)

