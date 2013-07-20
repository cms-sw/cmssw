import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/DTskim_cfg.py,v $'),
    annotation = cms.untracked.string('Collisions DT skim')
)

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(0),
    debugFlag = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(
#'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/bit40or41skim.root'
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/478/3466A37A-1FD9-DE11-9734-000423D94E1C.root'
'/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/123/151/2C0CB595-0EDE-DE11-921B-0030487C6062.root'
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/00F05467-F4DD-DE11-857C-003048D2C0F4.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/262EAE54-F9DD-DE11-B1DF-003048D37580.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/30484813-F5DD-DE11-A755-001D09F29619.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/3CE3F1C6-FADD-DE11-8AEA-001D09F251D1.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/464D32E3-FCDD-DE11-A9BD-001D09F28D54.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/6C8F0233-FCDD-DE11-BF8E-001D09F297EF.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/70F27BA6-F3DD-DE11-9988-000423D985B0.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/B06DF213-F5DD-DE11-A90B-001D09F253FC.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/C4897165-F4DD-DE11-A8FF-003048D375AA.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/C6AAB8AB-F3DD-DE11-BC66-001D09F2932B.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/CC8FB1A4-F3DD-DE11-9C83-000423D94E70.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/D0A1B0A8-F3DD-DE11-A151-000423D986C4.root',
#'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/FAF8E392-0EDE-DE11-87B2-001D09F24493.root'
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/314/02E8544C-70D8-DE11-85CF-001617C3B66C.root'
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/294/EADE90F7-4FD8-DE11-A235-000423D996C8.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/294/B2AD6AF5-4FD8-DE11-8562-001617C3B6DC.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/294/56043A15-52D8-DE11-B452-001D09F23A20.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/294/3E5BDB0C-52D8-DE11-B99D-001D09F29321.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/301/AC9D9BE7-5BD8-DE11-BFDC-000423D9870C.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/301/6A537973-5BD8-DE11-B339-001D09F2AF1E.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/301/3CA0854B-58D8-DE11-8BD8-001D09F25456.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/301/22110B42-5DD8-DE11-A842-001D09F2423B.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/301/1AB4C1B0-5BD8-DE11-9259-001D09F27067.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/301/0CEBC0A0-57D8-DE11-A764-000423D9890C.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/314/F62B040F-6CD8-DE11-9007-001D09F24664.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/314/EE7B1AC4-6CD8-DE11-97BB-0030487A1FEC.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/314/4CAB3B6C-6BD8-DE11-845C-000423D9890C.root',
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/122/314/02E8544C-70D8-DE11-85CF-001617C3B66C.root'
),
secondaryFileNames = cms.untracked.vstring()
)


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_P_V7::All"

process.muonDTDigis = cms.EDProducer("DTUnpackingModule",
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
     throw = cms.bool(False),    # throw exception on unknown path names
     saveTags = cms.bool(False)
)

process.HLTDTpath = cms.Path(process.HLTDT)


process.DTskim=cms.Path(process.muonDTDigis+process.hltDTActivityFilter)




process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/DTskim_cfg.py,v $'),
    annotation = cms.untracked.string('BSC skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)



process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('DTSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('DT_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('DTskim','HLTDTpath')
       )
)
process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.e = cms.EndPath(process.out)

