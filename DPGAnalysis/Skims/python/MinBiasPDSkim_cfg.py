import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.17 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/MinBiasPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Combined MinBias skim')
)

#
#
# This is for testing purposes.
#
#
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
# run 132440
'/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/26C8DED9-0E3C-DF11-9D83-0030487CD7B4.root'),
                           secondaryFileNames = cms.untracked.vstring(
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/CEF82055-F13B-DF11-BF11-000423D9989E.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/1CF54554-F13B-DF11-8BFB-000423D98BC4.root')
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V4::All' 

process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")
process.load('Configuration/EventContent/EventContent_cff')

process.FEVTEventContent.outputCommands.append('drop *_MEtoEDMConverter_*_*')

#
#  Load common sequences
#
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

###########################################################################################
#------------------------------------------
# parameters for the CSCSkim module
#------------------------------------------
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

#### the path
process.cscHaloSkim = cms.Path(process.hltBeamHalo+process.cscSkim)



#### output 
process.outputBeamHaloSkim = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTEventContent.outputCommands,
    fileName = cms.untracked.string("/tmp/malgeri/MinBiascscskimEvents.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('CSCSkim_BeamHalo_MinBias')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cscHaloSkim'))
)

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

process.load("DPGAnalysis.Skims.RPCRecHitFilter_cfi")
process.rpcRHSkim = cms.Path(process.RPCRecHitsFilter)
###########################################################################


########################## CSC Filter ############################
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



process.outputMuonSkim = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/malgeri/MuonSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('Muon_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("l1MuBitsSkim","dtHLTSkim","dtSkim","cscHLTSkim","cscSkimAlone","rpcRHSkim","rpcTecSkim","muonTracksSkim")
    )
)
####################################################################################
##################################good collisions############################################

process.L1T1coll=process.hltLevel1GTSeed.clone()
process.L1T1coll.L1TechTriggerSeeding = cms.bool(True)
process.L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')

process.l1tcollpath = cms.Path(process.L1T1coll)

process.primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)


process.noscraping = cms.EDFilter("FilterOutScraping",
applyfilter = cms.untracked.bool(True),
debugOn = cms.untracked.bool(False),
numtrack = cms.untracked.uint32(10),
thresh = cms.untracked.double(0.25)
)

process.goodvertex=cms.Path(process.primaryVertexFilter+process.noscraping)


process.collout = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/malgeri/good_coll.root'),
    outputCommands = process.FEVTEventContent.outputCommands,
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('GOODCOLL')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('goodvertex','l1tcollpath')
    )
)


##################################filter_rechit for ECAL############################################
process.load("DPGAnalysis.Skims.filterRecHits_cfi")

process.ecalrechitfilter = cms.Path(process.recHitEnergyFilter)


process.ecalrechitfilter_out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/malgeri/ecalrechitfilter.root'),
    outputCommands = process.FEVTEventContent.outputCommands,
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('ECALRECHIT')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('ecalrechitfilter')
    )
)

####################################################################################
##################################stoppedHSCP############################################


# this is for filtering on HLT path
process.hltstoppedhscp = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring("HLT_StoppedHSCP*"), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

process.HSCP=cms.Path(process.hltstoppedhscp)

process.outHSCP = cms.OutputModule("PoolOutputModule",
                               outputCommands =  process.FEVTEventContent.outputCommands,
                               fileName = cms.untracked.string('/tmp/malgeri/StoppedHSCP_filter.root'),
                               dataset = cms.untracked.PSet(
                                  dataTier = cms.untracked.string('RAW-RECO'),
                                  filterName = cms.untracked.string('Skim_StoppedHSCP')),
                               
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("HSCP")
    ))

##########################################################################################
#------------------------------------------
# parameters for the PFGCollisions skim (skim2)
#------------------------------------------

process.L1PFGcoll=process.hltLevel1GTSeed.clone()
process.L1PFGcoll.L1TechTriggerSeeding = cms.bool(True)
process.L1PFGcoll.L1SeedsLogicalExpression = cms.string('(36 OR 37 OR 38 OR 39 OR 40 OR 41 OR 42 OR 43 OR 8 OR 9 OR 10 OR 32 OR 33) AND 0')


process.pixprob = cms.EDFilter("FilterScrapingPixelProbability",
                                apply_filter                 = cms.untracked.bool( True  ),
                                select_collision             = cms.untracked.bool( True ),
                                select_pkam                  = cms.untracked.bool( False ),
                                select_other                 = cms.untracked.bool( False ),
                                low_probability              = cms.untracked.double( 0.0 ),
                                low_probability_fraction_cut = cms.untracked.double( 0.4 )
                                )

#### the path
process.pfgskim2_a = cms.Path(process.L1PFGcoll*process.pixprob*~process.L1T1coll*~process.primaryVertexFilter)
process.pfgskim2_b = cms.Path(process.L1PFGcoll*process.pixprob*~process.L1T1coll*~process.noscraping)

#### output 
process.outputpfgskim2 = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTEventContent.outputCommands,
    fileName = cms.untracked.string("/tmp/malgeri/PGFSkim2.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('PFGSkim2')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('pfgskim2_a','pfgskim2_b'))
)

###########################################################################################
#------------------------------------------
# parameters for the PFGCollisions skim (skim3)
#------------------------------------------

process.L1PFGbackcross=process.hltLevel1GTSeed.clone()
process.L1PFGbackcross.L1TechTriggerSeeding = cms.bool(True)
process.L1PFGbackcross.L1SeedsLogicalExpression = cms.string('0 AND (36 OR 37 OR 38 OR 39 OR 40 OR 41 OR 42 OR 43 OR 8 OR 9 OR 10 OR 32 OR 33)')

process.L1PFGbacknoncross=process.hltLevel1GTSeed.clone()
process.L1PFGbacknoncross.L1TechTriggerSeeding = cms.bool(True)
process.L1PFGbacknoncross.L1SeedsLogicalExpression = cms.string('NOT 0 AND NOT 7 AND (36 OR 37 OR 38 OR 39 OR 40 OR 41 OR 42 OR 43 OR 8 OR 9 OR 10 OR 32 OR 33)')


process.skimmingpixback = cms.EDFilter("FilterScrapingPixelProbability",
                                apply_filter                 = cms.untracked.bool( True  ),
                                select_collision             = cms.untracked.bool( False ),
                                select_pkam                  = cms.untracked.bool( True ),
                                select_other                 = cms.untracked.bool( True ),
                                low_probability              = cms.untracked.double( 0.0 ),
                                low_probability_fraction_cut = cms.untracked.double( 0.4 )
                                )


#### the path
process.pfgskim3cross = cms.Path(process.L1PFGbackcross*process.skimmingpixback)
process.pfgskim3noncross = cms.Path(process.L1PFGbacknoncross)



#### output 
process.outputpfgskim3 = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTEventContent.outputCommands,
    fileName = cms.untracked.string("/tmp/malgeri/Background.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('BEAMBKGV2')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('pfgskim3cross','pfgskim3noncross'))
)

###########################################################################################
###########################################################################################

#===========================================================

#################################logerrorharvester############################################
process.load("FWCore.Modules.logErrorFilter_cfi")

process.logerrorpath=cms.Path(process.logErrorFilter)

process.outlogerr = cms.OutputModule("PoolOutputModule",
                               outputCommands =  process.FEVTEventContent.outputCommands,
                               fileName = cms.untracked.string('/tmp/malgeri/logerror_filter.root'),
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

process.outpath = cms.EndPath(process.outputBeamHaloSkim+process.outputMuonSkim+process.collout+process.outHSCP+process.ecalrechitfilter_out+process.outputpfgskim2+process.outputpfgskim3+process.outlogerr)



 
