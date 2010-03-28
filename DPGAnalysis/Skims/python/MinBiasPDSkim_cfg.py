import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.11 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/MinBiasPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Combined MinBias skim')
)

#
#
# This is for testing purposes.
#
#
# run 123151 lumisection 14
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
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
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

##################################################DT skim###############################################
process.muonDTDigis = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    inputLabel = cms.InputTag('source'),
#    fedbyType = cms.untracked.bool(True),
# fedbytype is tracked in 353
    fedbyType = cms.bool(True),
    useStandardFEDid = cms.bool(True),
    dqmOnly = cms.bool(False),                       
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


process.hltDTActivityFilter = cms.EDFilter("HLTDTActivityFilter",
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

process.HLTDT =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_Activity_DT'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

process.HLTDTpath = cms.Path(process.HLTDT)
process.DTskim=cms.Path(process.muonDTDigis+process.hltDTActivityFilter)

process.DTskimout = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/malgeri/DTSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('DT_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('DTskim','HLTDTpath')
       )
)

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

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
##################################beam backg filter#################################################
process.L1T1bkgcross=process.hltLevel1GTSeed.clone()
process.L1T1bkgcross.L1TechTriggerSeeding = cms.bool(True)
process.L1T1bkgcross.L1SeedsLogicalExpression = cms.string('0 AND NOT (40 OR 41) AND ((36 OR 37 OR 38 OR 39) OR (42 AND NOT 43) OR (43 AND NOT 42))')

process.l1tbkgcrosspath = cms.Path(process.L1T1bkgcross)

process.L1T1bkgnocross=process.hltLevel1GTSeed.clone()
process.L1T1bkgnocross.L1TechTriggerSeeding = cms.bool(True)
process.L1T1bkgnocross.L1SeedsLogicalExpression = cms.string('NOT 0 AND NOT 7 AND (36 OR 37 OR 38 OR 39 OR 40 OR 41 OR 42 OR 43 OR 8 OR 9 OR 10)')

process.l1tbkgnocrosspath = cms.Path(process.L1T1bkgnocross)

process.bkgout = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/malgeri/bkg.root'),
    outputCommands = process.FEVTEventContent.outputCommands,
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('BEAMBKG')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('l1tbkgcrosspath','l1tbkgnocrosspath')
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



#===========================================================

process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.outputBeamHaloSkim+process.DTskimout+process.collout+process.bkgout+process.outHSCP+process.ecalrechitfilter_out)



 
