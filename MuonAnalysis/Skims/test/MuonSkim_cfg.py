import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONSKIM")


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V2P::All"

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source("PoolSource",
                            debugVerbosity = cms.untracked.uint32(0),
                            debugFlag = cms.untracked.bool(False),
                            fileNames = cms.untracked.vstring('file:/data/b/bellan/Run123592/RECO/E609699F-2BE2-DE11-A59D-003048D2C108.root',
                                                              'file:/data/b/bellan/Run123592/RECO/5C5983C5-2AE2-DE11-84A1-0019B9F72BAA.root'),
                            
                            secondaryFileNames = cms.untracked.vstring('file:/data/b/bellan/Run123592/RAW/56AA9B12-25E2-DE11-A226-001D09F290CE.root',
                                                                       'file:/data/b/bellan/Run123592/RAW/1C0F2100-23E2-DE11-B546-001617E30D4A.root',
                                                                       'file:/data/b/bellan/Run123592/RAW/1428BDB6-23E2-DE11-8E8B-001D09F2516D.root')
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/bsc_activity_cfg.py,v $'),
    annotation = cms.untracked.string('BSC skim')
    )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))


###################### DT Activity Filter ######################
process.muonDTDigis = cms.EDFilter("DTUnpackingModule",
    dataType = cms.string('DDU'),
    useStandardFEDid = cms.untracked.bool(True),
    fedbyType = cms.untracked.bool(True),
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    )
)


process.hltDTActivityFilter = cms.EDFilter( "HLTDTActivityFilter",
 inputDCC         = cms.InputTag( "dttfDigis" ),   
 inputDDU         = cms.InputTag( "muonDTDigis" ),   
 inputDigis       = cms.InputTag( "muonDTDigis" ),   
 processDCC       = cms.bool( False ),   
 processDDU       = cms.bool( False ),   
 processDigis     = cms.bool( True ),   
 processingMode   = cms.int32( 0 ),   # 0=(DCC | DDU) | Digis/ 
                                      # 1=(DCC & DDU) | Digis/
                                      # 2=(DCC | DDU) & Digis/
                                      # 3=(DCC & DDU) & Digis/   
 minChamberLayers = cms.int32( 6 ),
 maxStation       = cms.int32( 3 ),
 minQual          = cms.int32( 2 ),   # 0-1=L 2-3=H 4=LL 5=HL 6=HH/
 minDDUBX         = cms.int32( 9 ),
 maxDDUBX         = cms.int32( 14 ),
 minActiveChambs  = cms.int32( 1 )
)

# this is for filtering on HLT path
process.hltHighLevel = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_Activity_DT'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(True)    # throw exception on unknown path names
 )
process.dtSkim=cms.Path(process.hltHighLevel+process.muonDTDigis+process.hltDTActivityFilter)

###########################################################################


########################## RPC Activity Filter ############################
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load("HLTrigger.HLTfilters.hltLevel1GTSeed_cfi")
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('31')
process.rpcSkim = cms.Path(process.hltLevel1GTSeed)
###########################################################################


########################## CSC Filter ############################
process.load("DPGAnalysis.Skims.CSCSkim_cfi")
process.cscSkimP = cms.Path(process.cscSkim)
###########################################################################

########################## Muon tracks Filter ############################
process.load("MuonAnalysis.Skims.MuonSkim_cfi")
process.muonTracksSkim = cms.Path(process.muonSkim)
###########################################################################




process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('MuonSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('Muon_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("dtSkim","cscSkimP","rpcSkim","muonTracksSkim")
    )
)

process.e = cms.EndPath(process.out)

