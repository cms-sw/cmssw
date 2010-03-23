import FWCore.ParameterSet.Config as cms

### event content for heavy-ion analysis objects

jetContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep patJets_selected*_*_*'
      )
    )

trkContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_hiSelectedTracks_*_*',
      'keep *_hiPixel3PrimTracks_*_*' # low-fake selection on this collection?
      )
    )

muonContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep patMuons_selected*_*_*',
      )
    )

photonContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep patPhotons_selected*_*_*'
      )
    )

corrContent = cms.PSet(
    outputCommands = cms.untracked.vstring( #collections of RecoChargedCandidates
      'keep *_allTracks_*_*',
      'keep *_allPxltracks_*-*'
      )
    )

hiCommon = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
      'keep *_TriggerResults_*_*',
      'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
      'keep *_hiSelectedVertex_*_*'                              
      )
    )

### combinations for specific skims

jetTrkSkim = hiCommon.clone()
jetTrkSkim.outputCommands.extend(jetContent.outputCommands)
jetTrkSkim.outputCommands.extend(trkContent.outputCommands)

muonTrkSkim = hiCommon.clone()
muonTrkSkim.outputCommands.extend(muonContent.outputCommands)
muonTrkSkim.outputCommands.extend(trkContent.outputCommands)
