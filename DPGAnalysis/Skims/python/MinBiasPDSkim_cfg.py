import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.10 $'),
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
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/FE660B03-69E8-DE11-93F9-0019B9F72BAA.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/F2D1D72A-6BE8-DE11-A557-001D09F23A20.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/EC3C4DA9-67E8-DE11-A0C3-001D09F24934.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/E8C42B2A-6BE8-DE11-A36D-001D09F251FE.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/E6A991ED-6DE8-DE11-BA77-000423D94E70.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/E64455DE-64E8-DE11-9F6A-001D09F29321.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/DC39D802-69E8-DE11-9DCC-001D09F242EA.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/D8291FA9-67E8-DE11-899A-001D09F29538.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/D68F3E7A-63E8-DE11-A50C-001D09F29321.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/D4C291F7-66E8-DE11-A2C0-001D09F2525D.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/D25708F8-66E8-DE11-80A3-001D09F24DDF.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/CCAE67A4-6EE8-DE11-8C6F-001D09F24FEC.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/CAE73419-64E8-DE11-B6AA-001D09F25401.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/CA2A533D-66E8-DE11-8226-000423D9863C.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/C8F9D038-66E8-DE11-9492-001D09F252E9.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/C014435C-68E8-DE11-82CA-001D09F2527B.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/BE25A003-69E8-DE11-93BC-001D09F24493.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/BA1A63A4-6EE8-DE11-AE97-001D09F24303.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/B8FDDE3B-66E8-DE11-A009-000423D992A4.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/B21250A4-6EE8-DE11-BF8C-0019B9F70468.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/B0D8A32B-6BE8-DE11-80EC-001D09F29597.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/B04A6138-66E8-DE11-AE0B-001D09F25217.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/AE3C6FD2-69E8-DE11-8441-001D09F24EE3.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/AA960086-65E8-DE11-9220-001D09F2532F.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/A4C23F2A-6BE8-DE11-9A5A-001D09F231B0.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/A2365202-69E8-DE11-ABE9-001D09F2B2CF.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/A0470D7A-63E8-DE11-9F3E-001D09F29114.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/9C6837D7-69E8-DE11-A441-001D09F2545B.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/9813251B-64E8-DE11-9F76-001D09F2447F.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/94F87E3D-66E8-DE11-B005-001D09F24F1F.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/94F24AF4-66E8-DE11-B5AF-001617C3B778.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/8C7361A0-6CE8-DE11-9F67-000423D94C68.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/8AE33CA2-6CE8-DE11-B636-000423D99A8E.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/84C7355C-68E8-DE11-BDF1-001D09F2514F.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/84987E2B-6BE8-DE11-96C4-001D09F2462D.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/7ABC478A-6AE8-DE11-98E3-000423D992DC.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/78BF087A-63E8-DE11-A8E5-0019B9F72BAA.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/6EAC9C02-69E8-DE11-94EF-001D09F28F25.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/6404B7A3-6EE8-DE11-9F44-001D09F2423B.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/5067AE59-68E8-DE11-80FC-001617C3B6DE.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/4C8CD884-65E8-DE11-B515-001D09F231B0.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/44D5F288-6AE8-DE11-8D37-000423D99CEE.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/42BD7017-64E8-DE11-AA1E-001D09F24FBA.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/3CDEA82B-6BE8-DE11-A225-000423D6CA02.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/3A77C22A-6BE8-DE11-81A5-001D09F29533.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/38133B43-6DE8-DE11-B260-0019B9F72D71.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/2E11E19F-6CE8-DE11-A9EB-001617E30CE8.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/2C8D3B7A-63E8-DE11-9B56-001D09F291D2.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/26FC33CF-6BE8-DE11-8C83-0019DB29C5FC.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/26DC301D-64E8-DE11-A230-001D09F251BD.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/221E9B55-6FE8-DE11-BDDD-001D09F28D54.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/20463F19-64E8-DE11-99EF-001D09F231C9.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/1C8EF2EF-6DE8-DE11-B151-001D09F232B9.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/1AED53A1-62E8-DE11-B18D-001617C3B6CE.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/168B85DD-64E8-DE11-9C53-001D09F2910A.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/167C3F43-6DE8-DE11-BF89-001D09F28EA3.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/165F6E17-64E8-DE11-815E-001D09F24D4E.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/12156943-6DE8-DE11-BB2E-001D09F25041.root',
#'/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/124/120/04582C18-64E8-DE11-82C3-001D09F2546F.root')
'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/151/6ADC6A1B-01DE-DE11-8FBD-00304879FA4A.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/151/6ADC6A1B-01DE-DE11-8FBD-00304879FA4A.root'),
                            secondaryFileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/3CE3F1C6-FADD-DE11-8AEA-001D09F251D1.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/6C8F0233-FCDD-DE11-BF8E-001D09F297EF.root')
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
process.GlobalTag.globaltag = 'GR09_R_35X_V2::All' 

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
 minActiveChambs  = cms.int32( 1 )
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



 
