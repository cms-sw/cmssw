import FWCore.ParameterSet.Config as cms


process = cms.Process("makeSD")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('JetMETAOD and Tau central skim'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/test/CSmaker_JetMETAOD_Tau_PDJetMETTau_1e29_reprocess370_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR_R_37X_V6A::All"  


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/601/F85204EE-EB40-DF11-8F71-001A64789D1C.root'
        ),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/F6887FD0-9371-DE11-B69E-00304879FBB2.root'
        )
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

import HLTrigger.HLTfilters.hltHighLevelDev_cfi

### JetMET AOD CS
process.DiJetAve_1e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.DiJetAve_1e29.HLTPaths = ("HLT_DiJetAve15U_8E29","HLT_DiJetAve30U_8E29")
process.DiJetAve_1e29.HLTPathsPrescales  = cms.vuint32(1,1)
process.DiJetAve_1e29.HLTOverallPrescale = cms.uint32(1)
process.DiJetAve_1e29.andOr = True

process.filterCsDiJetAve_1e29 = cms.Path(process.DiJetAve_1e29)


### Tau skim CS
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff")
process.load("HLTrigger/HLTfilters/hltLevel1GTSeed_cfi")
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('(0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39))')

process.scrapping = cms.EDFilter("FilterOutScraping",
    	applyfilter = cms.untracked.bool(True),
    	debugOn = cms.untracked.bool(False),
        numtrack = cms.untracked.uint32(10),
        thresh = cms.untracked.double(0.25)
)

process.PFTausSelected = cms.EDFilter("PFTauSelector",
    src = cms.InputTag("shrinkingConePFTauProducer"),
    discriminators = cms.VPSet(
	cms.PSet( discriminator=cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"),
		   selectionCut=cms.double(0.5)
	),
    ),
    cut = cms.string('et > 15. && abs(eta) < 2.5') 
)

process.PFTauSkimmed = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag('PFTausSelected'),
  minNumber = cms.uint32(1)
)


process.tauFilter = cms.Path(
	process.hltLevel1GTSeed *
	process.scrapping *
	process.PFTausSelected *
	process.PFTauSkimmed
)



process.outputCsDiJet = cms.OutputModule("PoolOutputModule",
                                         dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('USER'),
        filterName = cms.untracked.string('CS_DiJetAve')),
                                         outputCommands = cms.untracked.vstring(
        'drop *',
        #------- CaloJet collections ------
        'keep recoCaloJets_kt4CaloJets_*_*',
        'keep recoCaloJets_kt6CaloJets_*_*',
        'keep recoCaloJets_ak5CaloJets_*_*',
        'keep recoCaloJets_ak7CaloJets_*_*',
        'keep recoCaloJets_iterativeCone5CaloJets_*_*',
        #------- CaloJet ID ---------------
        'keep *_kt4JetID_*_*',
        'keep *_kt6JetID_*_*',
        'keep *_ak5JetID_*_*',
        'keep *_ak7JetID_*_*',
        'keep *_ic5JetID_*_*', 
        #------- PFJet collections ------  
        'keep recoPFJets_kt4PFJets_*_*',
        'keep recoPFJets_kt6PFJets_*_*',
        'keep recoPFJets_ak5PFJets_*_*',
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
        'keep *_met_*_*'),
                                         SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterCsDiJetAve_1e29')), 
                                         fileName = cms.untracked.string('CS_JetAOD_DiJetAve_1e29.root')
                                         )

process.outputCsTau = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-RECO'),
        filterName = cms.untracked.string('CS_Tau')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('tauFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Tau_1e29.root')
                                        )


process.this_is_the_end = cms.EndPath(
process.outputCsDiJet +
process.outputCsTau
)
