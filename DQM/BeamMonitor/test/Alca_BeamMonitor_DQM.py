import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-v2/000/146/510/7AA2EFFA-A9C7-DF11-A332-001617C3B69C.root',
'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-v2/000/146/514/C8E7D08D-BBC7-DF11-9617-001617E30E28.root',
'/store/data/Run2010B/MinimumBias/ALCARECO/TkAlMinBias-v2/000/146/514/02FDBC55-30C8-DF11-BB4C-001617DBD224.root',
#'/store/relval/CMSSW_3_8_0/MinimumBias/ALCARECO/GR_R_38X_V7_RelVal_col_10_bs_TkAlMinBias-v1/0007/90B6F8D6-4E96-DF11-B3A9-0018F3D0970A.root',
#'/store/relval/CMSSW_3_8_0/MinimumBias/ALCARECO/GR_R_38X_V7_RelVal_col_10_bs_TkAlMinBias-v1/0007/8A6CDD40-C995-DF11-B6D5-0018F3D096AA.root'
    )
#    , duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories = ["AlcaBeamMonitor"]
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
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
#process.MessageLogger.statistics.append('cout')

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

process.load("RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi")
process.onlineBeamSpot = process.onlineBeamSpotProducer.clone()

process.AlcaBeamMonitor.TrackLabel = 'ALCARECOTkAlMinBias'
process.AlcaBeamMonitor.BeamFitter.TrackCollection = 'ALCARECOTkAlMinBias'

process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    								   record = cms.string('BeamSpotObjectsRcd'),			        
#    								   tag = cms.string('BeamSpotObjects_2009_LumiBased_v16_offline') 
    								   tag = cms.string('BeamSpotObject_ByLumi') 
    								  )
						         ),								        
                                         #connect = cms.string('frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT')
#                                         connect = cms.string('sqlite_file:promptCalibConditions_2.db')
                                         connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_PROMPT/ALCAPROMPTHarvest-Run146514-StreamExpress-ALCAPROMPT-305756.db')
#                                         connect = cms.string('sqlite_file:ALCAPROMPTHarvest-Run146514-StreamExpress-ALCAPROMPT-305756.db')
                                         #connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
                                         #connect = cms.string('frontier://PromptProd/CMS_COND_31X_BEAMSPOT')
                                        )

#process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","BeamSpotDBSource")


process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('AlcaBeamMonitorDQM.root'),
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
                      process.onlineBeamSpot*
                      process.AlcaBeamMonitor*
		      process.endOfProcess*
		      process.DQMoutput)
process.schedule = cms.Schedule(process.pp)
