import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDimuonReco")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.ZReco.dimuons_SkimPathsFromPatTuple_cff")

process.dimuonsOutputModule = cms.OutputModule(
    "PoolOutputModule",
     outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_selectedLayer1TrackCands_*_*',
    'keep *_dimuons_*_*', 
    'keep *_dimuonsOneTrack_*_*', 
    'keep *_dimuonsGlobal_*_*', 
    'keep *_dimuonsOneStandAloneMuon_*_*', 
    'keep *_muonMatch_*_*', 
    'keep *_trackMuMatch_*_*', 
    'keep *_allDimuonsMCMatch_*_*'
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        'dimuonsPath',
        'dimuonsOneTrackPath'
      )
    ),
    dataset = cms.untracked.PSet(
      filterName = cms.untracked.string('dimuon'),
      dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('file:dimuons_pattuple.root')
)

    ## std pat layer1 event content
process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")
process.dimuonsOutputModule.outputCommands.extend(process.patLayer1EventContent.outputCommands)
    
    ## additionally reco'ed pat layer1 event content
process.patTupleEventContent_pat = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_selectedLayer1CaloTaus_*_*' ## reco'd caloTaus
                                           )
)
process.dimuonsOutputModule.outputCommands.extend(process.patTupleEventContent_pat.outputCommands)
        
    ## additional aod event content
process.patTupleEventContent_aod = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_genParticles_*_*',          ## all genPaticles (unpruned)
      'keep *_genEventScale_*_*',         ## genEvent info
      'keep *_genEventWeight_*_*',        ## genEvent info
      'keep *_genEventProcID_*_*',        ## genEvent info
      'keep *_genEventPdfInfo_*_*',       ## genEvent info
      'keep *_hltTriggerSummaryAOD_*_*',  ## hlt TriggerEvent
      'keep *_towerMaker_*_*',            ## all caloTowers
      'keep *_offlineBeamSpot_*_*'        ## offline beamspot
    )
)
process.dimuonsOutputModule.outputCommands.extend(process.patTupleEventContent_aod.outputCommands)


process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)
                                                  
process.maxEvents = cms.untracked.PSet(
  input =cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
       'file:/afs/cern.ch/user/h/hegner/scratch0/PAT/testPatTuple_recHits_221.root'
  )
)

process.endp = cms.EndPath(
  process.dimuonsOutputModule
)


