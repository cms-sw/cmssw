import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/CommPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Combined Commissioning skim')
)

# efficiency on 1000 input events
# file:/tmp/malgeri/COMM_MuonDPGSkim.root
# /tmp/azzi/COMM_MuonDPGSkim.root ( 271 events, 118916076 bytes )

#
#
# This is for testing purposes.
#
#
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/data/Run2010A/Commissioning/RECO/v1/000/136/066/244E5A07-9466-DF11-9091-000423D95220.root'),
                           secondaryFileNames = cms.untracked.vstring('/store/data/Run2010A/Commissioning/RAW/v1/000/136/066/28D89EF5-3C66-DF11-9AAA-0019B9F705A3.root')
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V8::All' 

process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")
process.load('Configuration/EventContent/EventContent_cff')

#drop collections created on the fly
process.FEVTEventContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
process.FEVTEventContent.outputCommands.append("drop *_*_*_SKIM")

#
#  Load common sequences
#
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')


###################### DT Activity Filter ######################

from EventFilter.DTRawToDigi.dtunpackerDDUGlobal_cfi import dtunpacker

process.muonDTDigis = dtunpacker.clone()

process.hltDTActivityFilter = cms.EDFilter( "HLTDTActivityFilter",
    inputDCC         = cms.InputTag( "dttfDigis" ),
    inputDDU         = cms.InputTag( "muonDTDigis" ),
    inputRPC         = cms.InputTag( "hltGtDigis" ),
    inputDigis       = cms.InputTag( "muonDTDigis" ),
    processDCC       = cms.bool( False ),
    processDDU       = cms.bool( False ),
    processRPC       = cms.bool( False ),
    processDigis     = cms.bool( True ),

    maxDeltaPhi = cms.double( 1.0 ),
    maxDeltaEta = cms.double( 0.3 ),

    orTPG         = cms.bool( True ),
    orRPC         = cms.bool( True ),
    orDigi        = cms.bool( True ),

    minChamberLayers = cms.int32( 5 ),
    maxStation       = cms.int32( 3 ),
    minTPGQual       = cms.int32( 2 ),   # 0-1=L 2-3=H 4=LL 5=HL 6=HH
    minDDUBX         = cms.int32( 8 ),
    maxDDUBX         = cms.int32( 13 ),
    minDCCBX         = cms.int32( -1 ),
    maxDCCBX         = cms.int32( 1 ),
    minRPCBX         = cms.int32( -1 ),
    maxRPCBX         = cms.int32( 1 ),
    minActiveChambs  = cms.int32( 1 ),
    activeSectors    = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12),
    saveTags = cms.bool(False)
                                     )

# this is for filtering on HLT path
process.HLTDT =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_Activity_DT', 'HLT_Activity_DT_Tuned'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False),    # throw exception on unknown path names
     saveTags = cms.bool(False)
)

process.dtHLTSkim = cms.Path(process.HLTDT)

process.dtSkim=cms.Path(process.muonDTDigis+process.hltDTActivityFilter)


############################ L1 Muon bits #################################

process.l1RequestPhAlgos = process.hltLevel1GTSeed.clone()
# False allows to read directly from L1 instead fo candidate ObjectMap
process.l1RequestPhAlgos.L1UseL1TriggerObjectMaps = cms.bool(False)
    #
    # option used forL1UseL1TriggerObjectMaps = False only
    # number of BxInEvent: 1: L1A=0; 3: -1, L1A=0, 1; 5: -2, -1, L1A=0, 1, 
# online is used 5
process.l1RequestPhAlgos.L1NrBxInEvent = cms.int32(5)

# Request the or of the following bits: from 54 to 62 and 106-107

process.l1RequestPhAlgos.L1SeedsLogicalExpression = cms.string(
    'L1_SingleMuBeamHalo OR L1_SingleMuOpen OR L1_SingleMu0 OR L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_SingleMu10 OR L1_SingleMu14 OR L1_SingleMu20 OR L1_DoubleMuOpen OR L1_DoubleMu3')

process.l1MuBitsSkim = cms.Path(process.l1RequestPhAlgos)

###########################################################################


########################## RPC Filters ############################

process.l1RequestTecAlgos = process.hltLevel1GTSeed.clone()

process.l1RequestTecAlgos.L1TechTriggerSeeding = cms.bool(True)
process.l1RequestTecAlgos.L1SeedsLogicalExpression = cms.string('31')

process.rpcTecSkim = cms.Path(process.l1RequestTecAlgos)

###########################################################################
########################## CSC Filter ############################

process.load("DPGAnalysis/Skims/CSCSkim_cfi")
#set to minimum activity
process.cscSkim.minimumSegments = 1
process.cscSkim.minimumHitChambers = 1

# this is for filtering on HLT path
process.hltBeamHalo = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_CSCBeamHalo','HLT_CSCBeamHaloOverlapRing1','HLT_CSCBeamHaloOverlapRing','HLT_CSCBeamHaloRing2or3'), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

# path already defined above
#### the paths (single paths wrt combined paths above)
process.cscHLTSkim = cms.Path(process.hltBeamHalo)
process.cscSkimAlone = cms.Path(process.cscSkim)
###########################################################################

########################## Muon tracks Filter ############################
process.muonSkim=cms.EDFilter("CandViewCountFilter", 
                 src =cms.InputTag("muons"), minNumber = cms.uint32(1))
process.muonTracksSkim = cms.Path(process.muonSkim)


###########################################################################



process.outputMuonDPGSkim = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/azzi/COMM_MuonDPGSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('MuonDPG_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("l1MuBitsSkim","dtHLTSkim","dtSkim","cscHLTSkim","cscSkimAlone","rpcTecSkim","muonTracksSkim")
    )
)
####################################################################################

#################################logerrorharvester############################################
process.load("FWCore.Modules.logErrorFilter_cfi")
from Configuration.StandardSequences.RawToDigi_Data_cff import gtEvmDigis

process.gtEvmDigis = gtEvmDigis.clone()
process.stableBeam = cms.EDFilter("HLTBeamModeFilter",
                                  L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                                  AllowedBeamMode = cms.vuint32(11),
                                  saveTags = cms.bool(False)
                                  )

process.logerrorpath=cms.Path(process.gtEvmDigis+process.stableBeam+process.logErrorFilter)

process.outlogerr = cms.OutputModule("PoolOutputModule",
                               outputCommands =  process.FEVTEventContent.outputCommands,
                               fileName = cms.untracked.string('/tmp/azzi/logerror_filter.root'),
                               dataset = cms.untracked.PSet(
                                  dataTier = cms.untracked.string('RAW-RECO'),
                                  filterName = cms.untracked.string('Skim_logerror')),
                               
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("logerrorpath")
    ))

#===========================================================


process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.outputMuonDPGSkim+process.outlogerr)



 
