import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
DiJetAveSkim_Trigger = hlt.triggerResultsFilter.clone()
DiJetAveSkim_Trigger.triggerConditions = cms.vstring('HLT_DiJetAve*',)
DiJetAveSkim_Trigger.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
DiJetAveSkim_Trigger.l1tResults = cms.InputTag("")
DiJetAveSkim_Trigger.throw = cms.bool( False )


#event content
DiJetAveSkim_EventContent = cms.PSet(
     outputCommands = cms.untracked.vstring(
        'drop *',
        #------- CaloJet collections ------
        'keep recoCaloJets_kt4CaloJets_*_*',
        'keep recoCaloJets_kt6CaloJets_*_*',
        'keep recoCaloJets_ak4CaloJets_*_*',
        'keep recoCaloJets_ak7CaloJets_*_*',
        'keep recoCaloJets_iterativeCone5CaloJets_*_*',
        #------- CaloJet ID ---------------
        'keep *_kt4JetID_*_*',
        'keep *_kt6JetID_*_*',
        'keep *_ak4JetID_*_*',
        'keep *_ak7JetID_*_*',
        'keep *_ic5JetID_*_*',
        #------- PFJet collections ------
        'keep recoPFJets_kt4PFJets_*_*',
        'keep recoPFJets_kt6PFJets_*_*',
        'keep recoPFJets_ak4PFJets_*_*',
        'keep recoPFJets_ak7PFJets_*_*',
        'keep recoPFJets_iterativeCone5PFJets_*_*',
        #------- JPTJet collections ------
        'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*',
        #'keep *_ak4JPTJets_*_*',
        #'keep *_iterativeCone5JPTJets_*_*',
        #------- Trigger collections ------
        'keep edmTriggerResults_TriggerResults_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep L1GlobalTriggerObjectMapRecord_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        #------- Tracks collection --------
        'keep recoTracks_generalTracks_*_*',
        #------- CaloTower collection -----
        'keep *_towerMaker_*_*',
        #------- Various collections ------
        'keep *_EventAuxilary_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_hcalnoise_*_*',
        #------- MET collections ----------
        'keep *_metHO_*_*',
        'keep *_metNoHF_*_*',
        'keep *_metNoHFHO_*_*',
        'keep *_met_*_*')
)

