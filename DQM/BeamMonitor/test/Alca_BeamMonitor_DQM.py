import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/FCA048AB-4C4F-E011-8555-003048F1C58C.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/FA514098-4A4F-E011-ADAA-0030487CD13A.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/F89CE0F8-4B4F-E011-8346-003048F1182E.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/D855D8F9-4B4F-E011-A37B-003048F024E0.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/D4D56CFD-4B4F-E011-ABB1-003048F118E0.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/CE2ACEF7-4B4F-E011-B274-0030487C6090.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/CCE39EBC-4A4F-E011-B44B-0030487BC68E.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/C8C63CF8-4B4F-E011-9C74-003048F1BF66.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/C2D88A37-494F-E011-BE4C-003048F1C420.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/B0D2DEF8-4B4F-E011-B0DE-003048F11C28.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/B042C396-4A4F-E011-A9CD-0030487A1FEC.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/AEA74AE4-494F-E011-BB1E-000423D33970.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/967AC8A0-4A4F-E011-A564-0030487C90D4.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/94F807B3-4A4F-E011-A47B-0030487CD704.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/92D3ECF8-4B4F-E011-8D03-003048F1C836.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/8C5DF496-4A4F-E011-B32A-003048F1C58C.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/8C30042E-494F-E011-880B-003048CFB40C.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/80AD139D-4A4F-E011-98FA-003048F118E0.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/76BD8436-494F-E011-BEB7-003048F1C424.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/76026696-4A4F-E011-927F-0030487C5CE2.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/6A2E65B8-4A4F-E011-A977-0030487A195C.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/6437D7B1-4A4F-E011-B09A-003048F1C836.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/5C84A6B6-4A4F-E011-89A6-0030487CD180.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/584AF8B0-4B4F-E011-9D63-0030487CD13A.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/4CE97EA2-4A4F-E011-95C6-0030487C912E.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/4CE2E1B4-4A4F-E011-86EC-0030487C6090.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/3C24DBE9-494F-E011-8603-0030487C7828.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/34C5BAA7-4A4F-E011-8502-000423D9890C.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/34AA912E-494F-E011-AA98-0030487CD13A.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/303205B1-4C4F-E011-AC44-003048F0258C.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/22F10A9D-4A4F-E011-8C51-0030487A18A4.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/1CA95DA7-4A4F-E011-9C5B-000423D9997E.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/1245CC9C-4A4F-E011-90DA-0030487CD7EA.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/0C8305B3-4A4F-E011-83E7-0030487A3232.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/080C69B6-4A4F-E011-9D5A-0030487CD7E0.root',
'/store/data/Run2011A/MinimumBias/RECO/PromptReco-v1/000/160/410/02939897-4A4F-E011-82A3-003048F118C6.root',

#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-v2/000/146/510/7AA2EFFA-A9C7-DF11-A332-001617C3B69C.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0068/B866C769-441B-E011-8CD1-0018F3D09704.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0062/C4AA5585-FA1A-E011-BD3B-003048678B1A.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0060/6A05E181-E51A-E011-9AD4-0026189438DD.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0068/2EA3886D-451B-E011-B06B-001A92810ADC.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0066/30DB81E5-2B1B-E011-9A57-00261894392D.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0063/201CD781-041B-E011-9B12-002618943836.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0062/F818CA1B-FD1A-E011-ABD3-0026189438B8.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-Dec22ReReco_v1/0060/18B1105A-E81A-E011-86E4-00261894389C.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-v2/000/146/514/C8E7D08D-BBC7-DF11-9617-001617E30E28.root',
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-v2/000/146/514/02FDBC55-30C8-DF11-BB4C-001617DBD224.root',
#'/store/relval/CMSSW_3_8_0/MinimumBias/ALCARECO/GR_R_38X_V7_RelVal_col_10_bs_TkAlMinBias-v1/0007/90B6F8D6-4E96-DF11-B3A9-0018F3D0970A.root',
#'/store/relval/CMSSW_3_8_0/MinimumBias/ALCARECO/GR_R_38X_V7_RelVal_col_10_bs_TkAlMinBias-v1/0007/8A6CDD40-C995-DF11-B6D5-0018F3D096AA.root'
    )
#    , duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(0)
    ),
    AlcaBeamMonitor = cms.untracked.PSet(
        #reportEvery = cms.untracked.int32(100) # every 1000th only
	limit = cms.untracked.int32(0)
    )
)
#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# this is for filtering on L1 technical trigger bit
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
#process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
#process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
#process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 ) AND NOT (36 OR 37 OR 38 OR 39)')
                                                                                                                                         
#process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'GR_R_38X_V9::All' #'GR_R_35X_V8::All'

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.load("DQM.BeamMonitor.AlcaBeamMonitor_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.AlcaBeamMonitor.TrackLabel = 'ALCARECOTkAlMinBias'
#process.AlcaBeamMonitor.BeamFitter.TrackCollection = 'ALCARECOTkAlMinBias'

process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    								   record = cms.string('BeamSpotObjectsRcd'),			        
#    								   tag = cms.string('BeamSpotObjects_2009_LumiBased_SigmaZ_v19_offline') 
    								   tag = cms.string('BeamSpotObject_ByLumi') 
    								  )
						         ),								        
#                                          connect = cms.string('frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT')
                                         connect = cms.string('sqlite_file:ALCAPROMPTHarvest-Run160410-StreamExpress-ALCAPROMPT-744882.db')
#                                         connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_PROMPT/ALCAPROMPTHarvest-Run146514-StreamExpress-ALCAPROMPT-305756.db')
#                                         connect = cms.string('sqlite_file:Payloads_568825468682397_641212847489195_BeamSpotObjects_2009_LumiBased_SigmaZ_v20_offline@1de2dc56-4500-11e0-a46a-003048d2bf8c.db')
                                         #connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
                                         #connect = cms.string('frontier://PromptProd/CMS_COND_31X_BEAMSPOT')
                                        )

#process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","BeamSpotDBSource")


process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('AlcaBeamMonitorDQM_160410.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

process.DQMoutput.outputCommands.append('drop *')
process.DQMoutput.outputCommands.append('keep *_MEtoEDMConverter_*_*')

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.pp = cms.Path(#process.hltLevel1GTSeed*
                      process.AlcaBeamMonitor
		      *process.endOfProcess
		      )
process.ep = cms.EndPath(process.DQMoutput)
#process.schedule = cms.Schedule(process.pp)
