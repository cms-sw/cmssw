import FWCore.ParameterSet.Config as cms
#input tag for HLT info in nanodst TriggerResults::HLT
l1NtupleProducer = cms.EDAnalyzer("L1NtupleProducer",
    verbose              = cms.untracked.bool(False),
    physVal              = cms.bool(True),
    generatorSource      = cms.InputTag("none"),
    simulationSource     = cms.InputTag("none"),
    hltSource            = cms.InputTag("TriggerResults::HLT"),
    gmtSource            = cms.InputTag("none"),
    gtEvmSource          = cms.InputTag("none"),
    gtSource             = cms.InputTag("hltGtDigis"),
    gctCentralJetsSource = cms.InputTag("none",""),
    gctNonIsoEmSource    = cms.InputTag("none",""),
    gctForwardJetsSource = cms.InputTag("none",""),
    gctIsoEmSource       = cms.InputTag("none",""),
    gctEnergySumsSource  = cms.InputTag("none",""),
    gctTauJetsSource     = cms.InputTag("none",""),
    gctIsoTauJetsSource  = cms.InputTag("none",""),
    rctSource            = cms.InputTag("none"),
    dttfSource           = cms.InputTag("none"),
    ecalSource           = cms.InputTag("none"),
    hcalSource           = cms.InputTag("none"),
    csctfTrkSource       = cms.InputTag("none"),
    csctfLCTSource       = cms.InputTag("none"),
    csctfStatusSource    = cms.InputTag("none"),
    csctfDTStubsSource   = cms.InputTag("none"),
    # if initCSCTFPtLutsPSet is True, then the CSCTF ptLUTs
    # get initialized from the PSet, csctfPtLutsPSet
    # useful for experts to overwrite the csctf pt LUTs
    initCSCTFPtLutsPSet  = cms.bool(False),
    csctfPtLutsPSet      = cms.PSet(LowQualityFlag = cms.untracked.uint32(4),
                                    ReadPtLUT = cms.bool(False),
                                    isBinary  = cms.bool(False),
                                    PtMethod = cms.untracked.uint32(32)
                                    ),
    maxRPC      	 = cms.uint32(12),
    maxDTBX     	 = cms.uint32(12),
    maxCSC      	 = cms.uint32(12),
    maxGMT      	 = cms.uint32(12),
    maxGT       	 = cms.uint32(12),
    maxRCTREG   	 = cms.uint32(400),
    maxDTPH     	 = cms.uint32(50),
    maxDTTH     	 = cms.uint32(50),
    maxDTTR     	 = cms.uint32(50),
    maxGEN      	 = cms.uint32(20),
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

