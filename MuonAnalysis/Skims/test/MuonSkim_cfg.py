import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONSKIM")

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR10_P_V4::All"

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
# run 124120
# run 124120 lumi section <40
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/E61F175C-C337-DF11-AC54-00261894393D.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/E0D21B99-C537-DF11-9A0B-0026189438BF.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/D0A74E6B-C937-DF11-B25E-002618FDA211.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/C42B2E73-D337-DF11-95D1-001A928116B0.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/A827FD5B-C337-DF11-949D-0026189438CF.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/A4E4B175-C837-DF11-AE54-00261894396D.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/8AE7496C-C937-DF11-B9A3-002618943863.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/36BFC724-DD37-DF11-960B-0018F3D0965A.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Mar24thReReco_PreProduction_v2/0101/00CA545F-C337-DF11-AF31-002618FDA25B.root'),
                           secondaryFileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/F6ADE109-6BE8-DE11-9680-000423D991D4.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/ECF0E939-68E8-DE11-A59D-003048D2C1C4.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/E2071E9D-6EE8-DE11-AD98-0016177CA7A0.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/DC0FA50D-6BE8-DE11-8A92-000423D94E70.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/BCDF0152-6FE8-DE11-A0F1-000423D986C4.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/AE6B4236-6DE8-DE11-8C73-001D09F2512C.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/6E2A54FD-74E8-DE11-B9BC-0030487C5CFA.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/5CED3D29-72E8-DE11-89BA-001D09F23C73.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/54E4CD5C-6AE8-DE11-9CC3-000423D99A8E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/4A01877F-6CE8-DE11-8CA7-000423DD2F34.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/1CBED2C8-70E8-DE11-A173-001D09F29533.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/1C6E94B1-75E8-DE11-9F9E-0030487D1BCC.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/0E586CFE-6FE8-DE11-90CB-001617C3B6C6.root')
#                             fileNames = cms.untracked.vstring('file:/tmp/bellan/E609699F-2BE2-DE11-A59D-003048D2C108.root',
#                                                               'file:/tmp/bellan/5C5983C5-2AE2-DE11-84A1-0019B9F72BAA.root'),
                            
#                             secondaryFileNames = cms.untracked.vstring('file:/tmp/bellan/56AA9B12-25E2-DE11-A226-001D09F290CE.root',
#                                                                        'file:/tmp/bellan/1C0F2100-23E2-DE11-B546-001617E30D4A.root',
#                                                                        'file:/tmp/bellan/1428BDB6-23E2-DE11-8E8B-001D09F2516D.root')
                            )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/MuonAnalysis/Skims/test/MuonSkim_cfg.py,v $'),
    annotation = cms.untracked.string('BSC skim')
    )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))


###################### DT Activity Filter ######################

from EventFilter.DTRawToDigi.dtunpackerDDUGlobal_cfi import dtunpacker

process.muonDTDigis = dtunpacker.clone()

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
 minActiveChambs  = cms.int32( 1 ),
 activeSectors    = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12)                                           
)

# this is for filtering on HLT path
process.HLTDT =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_Activity_DT'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

process.dtHLTSkim = cms.Path(process.HLTDT)

process.dtSkim=cms.Path(process.muonDTDigis+process.hltDTActivityFilter)


###########################################################################

from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed

############################ L1 Muon bits #################################

process.l1RequestPhAlgos = hltLevel1GTSeed.clone()
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

process.l1RequestTecAlgos = hltLevel1GTSeed.clone()

process.l1RequestTecAlgos.L1TechTriggerSeeding = cms.bool(True)
process.l1RequestTecAlgos.L1SeedsLogicalExpression = cms.string('31')
process.rpcTecSkim = cms.Path(process.l1RequestTecAlgos)

process.load("DPGAnalysis.Skims.RPCRecHitFilter_cfi")
process.rpcRHSkim = cms.Path(process.RPCRecHitsFilter)
###########################################################################


########################## CSC Filter ############################
from DPGAnalysis.Skims.CSCSkim_cfi import cscSkim

process.cscSkimLower = cscSkim.clone()

#set to minimum activity
process.cscSkimLower.minimumSegments = 1
process.cscSkimLower.minimumHitChambers = 1

# this is for filtering on HLT path
process.cscHLTBeamHaloSkim  = cms.EDFilter("HLTHighLevel",
                                   TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                                   # provide list of HLT paths (or patterns) you want
                                   HLTPaths = cms.vstring('HLT_CSCBeamHalo','HLT_CSCBeamHaloOverlapRing1','HLT_CSCBeamHaloOverlapRing','HLT_CSCBeamHaloRing2or3'), 
                                   eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
                                   andOr = cms.bool(True),    # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
                                   throw = cms.bool(False)    # throw exception on unknown path names
                                   )

#### the paths
process.cscHLTSkim = cms.Path(process.cscHLTBeamHaloSkim)

process.cscSkim = cms.Path(process.cscSkimLower)
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
        SelectEvents = cms.vstring("l1MuBitsSkim","dtHLTSkim","dtSkim","cscHLTSkim","cscSkim","rpcRHSkim","rpcTecSkim","muonTracksSkim")
    )
)

process.e = cms.EndPath(process.out)

