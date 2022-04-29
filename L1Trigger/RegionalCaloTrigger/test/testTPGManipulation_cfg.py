import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    'file:/scratch/bachtis/test.root'
       
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)

# standard includes
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_30X::All"


# unpack raw data
process.load("Configuration.StandardSequences.RawToDigi_cff")

# run trigger primitive generation on unpacked digis, then central L1
process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")

process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
process.simHcalTriggerPrimitiveDigis.inputLabel = 'hcalDigis'



process.rctTPGDigis = cms.EDProducer("L1RCTTPGProvider",
                          ecalTPGs = cms.InputTag("simEcalTriggerPrimitiveDigis"),
                          hcalTPGs = cms.InputTag("simHcalTriggerPrimitiveDigis"),
                          useECALCosmicTiming = cms.bool(False),
                          useHCALCosmicTiming = cms.bool(False),
                          preSamples = cms.int32(0),
                          postSamples = cms.int32(0),
                          HFShift = cms.int32(0),
                          HBShift = cms.int32(0)
                          )

process.rctDigis = cms.EDProducer("L1RCTProducer",
    ecalDigis = cms.VInputTag(
                              cms.InputTag("rctTPGDigis","ECALBxminus1"),
                              cms.InputTag("rctTPGDigis","ECALBx0"),
                              cms.InputTag("rctTPGDigis","ECALBxplus1")
    ),


    useDebugTpgScales = cms.bool(False),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    hcalDigis = cms.VInputTag(
                cms.InputTag("rctTPGDigis","HCALBxminus1"),
                cms.InputTag("rctTPGDigis","HCALBx0"),
                cms.InputTag("rctTPGDigis","HCALBxplus1")
    ),
    BunchCrossings = cms.vint32(-1,0,1)                      


)


process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")


process.L1Analysis = cms.EDAnalyzer("L1RCTTestAnalyzer",
    hcalDigisLabel = cms.InputTag("rctTPGDigis","HCALBx0"),
    showEmCands = cms.untracked.bool(True),
    ecalDigisLabel = cms.InputTag("rctTPGDigis","ECALBx0"),
    rctDigisLabel = cms.InputTag("rctDigis"),
    showRegionSums = cms.untracked.bool(False)
)


process.TFileService = cms.Service("TFileService",
                                 fileName = cms.string("histo.root"),
                                 closeFileFast = cms.untracked.bool(True)
                             )



# L1 configuration
process.load('L1Trigger.Configuration.L1DummyConfig_cff')


process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_rct*_*_*'),

    fileName = cms.untracked.string('L1output.root')
)




process.p = cms.Path(
    process.ecalDigis
    *process.hcalDigis
    *process.rctTPGDigis
    *process.rctDigis
    *process.L1Analysis
    
)

process.e = cms.EndPath(process.output)
process.schedule = cms.Schedule(process.p,process.e)
