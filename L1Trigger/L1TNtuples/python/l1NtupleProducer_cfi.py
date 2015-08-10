import FWCore.ParameterSet.Config as cms

l1NtupleProducer = cms.EDAnalyzer("L1NtupleProducer",
    verbose              = cms.untracked.bool(False),
    physVal              = cms.bool(True),

    generatorSource      = cms.InputTag("none"),
    simulationSource     = cms.InputTag("none"),
    hltSource            = cms.InputTag("TriggerResults::HLT"),
    gmtSource            = cms.InputTag("gtDigis"),
    gtEvmSource          = cms.InputTag("none"),
    gtSource             = cms.InputTag("gtDigis"),
    gctCentralJetsSource = cms.InputTag("gctDigis","cenJets"),
    gctNonIsoEmSource    = cms.InputTag("gctDigis","nonIsoEm"),
    gctForwardJetsSource = cms.InputTag("gctDigis","forJets"),
    gctIsoEmSource       = cms.InputTag("gctDigis","isoEm"),
    gctEnergySumsSource  = cms.InputTag("gctDigis",""),
    gctTauJetsSource     = cms.InputTag("gctDigis","tauJets"),
    gctIsoTauJetsSource  = cms.InputTag("gctDigis","isoTauJets"),  ## replace "gctDigis" with "none" when running Legacy to avoid annoying warning messages
    rctSource            = cms.InputTag("gctDigis"),
    dttfSource           = cms.InputTag("dttfDigis"),
    ecalSource           = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
    hcalSource           = cms.InputTag("hcalDigis"),
    csctfTrkSource       = cms.InputTag("csctfDigis"),
    csctfLCTSource       = cms.InputTag("csctfDigis"),
    csctfStatusSource    = cms.InputTag("csctfDigis"),
    csctfDTStubsSource   = cms.InputTag("csctfDigis:DT"),
    # if initCSCTFPtLutsPSet is True, then the CSCTF ptLUTs
    # get initialized from the PSet, csctfPtLutsPSet
    # useful for experts to overwrite the csctf pt LUTs
    initCSCTFPtLutsPSet  = cms.bool(False),
    csctfPtLutsPSet      = cms.PSet(LowQualityFlag = cms.untracked.uint32(4),
                                    ReadPtLUT = cms.bool(False),
                                    isBinary  = cms.bool(False),
                                    PtMethod = cms.untracked.uint32(32)
                                    ),

    maxRPC               = cms.uint32(12),
    maxDTBX              = cms.uint32(12),
    maxCSC               = cms.uint32(12),
    maxGMT               = cms.uint32(12),
    maxGT                = cms.uint32(12),
    maxRCTREG            = cms.uint32(400),
    maxDTPH              = cms.uint32(50),
    maxDTTH              = cms.uint32(50),
    maxDTTR              = cms.uint32(50),
    maxGEN               = cms.uint32(20),
    maxCSCTFTR           = cms.uint32(50),
    maxCSCTFLCTSTR       = cms.uint32(4),
    maxCSCTFLCTS         = cms.uint32(360),
    maxCSCTFSPS          = cms.uint32(12),

    puMCFile             = cms.untracked.string(""),
    puDataFile           = cms.untracked.string(""),
    puMCHist             = cms.untracked.string(""),
    puDataHist           = cms.untracked.string(""),

    useAvgVtx            = cms.untracked.bool(True),
    maxAllowedWeight     = cms.untracked.double(-1)                         
)

