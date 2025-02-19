import FWCore.ParameterSet.Config as cms


process = cms.Process("makeSD")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Tau central skim'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Skimming/test/CSmaker_Tau_PDTau_35e29_cfg.py,v $')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.GlobalTag.globaltag = "GR10_P_V7::All"  


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




process.outputCsTau = cms.OutputModule("PoolOutputModule",
                                        dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-RECO'),
        filterName = cms.untracked.string('CS_Tau')),
                                        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('tauFilter')),                                        
                                        outputCommands = process.FEVTEventContent.outputCommands,
                                        fileName = cms.untracked.string('CS_Tau_1e28.root')
                                        )


process.this_is_the_end = cms.EndPath(
process.outputCsTau
)
