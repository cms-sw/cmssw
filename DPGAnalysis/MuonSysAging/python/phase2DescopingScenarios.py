import FWCore.ParameterSet.Config as cms

def cscAging(process):
    # CSC aging
    
    process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
            csc2DRecHitsOverload = cms.PSet(
            initialSeed = cms.untracked.uint32(81)
            ),
    )
    
    process.csc2DRecHitsOverload = cms.EDProducer('CFEBBufferOverloadProducer',
                                                  cscRecHitTag = cms.InputTag("csc2DRecHits"),
                                                  failureRate = cms.untracked.double(0.15),
                                                  doUniformFailure = cms.untracked.bool(True),
                                                  doCFEBFailure = cms.untracked.bool(True),
                                                  )
    
    # change input to cscSegments
    process.cscSegments.inputObjects = "csc2DRecHitsOverload"

    # make a new collection of reduced rechits and feed those into the csc segment producer
    process.csclocalreco = cms.Sequence(
        process.csc2DRecHits+
        process.csc2DRecHitsOverload+
        process.cscSegments
    )
    return process

def hltCscAging(process):
    # CSC aging
    
    process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                       hltCsc2DRecHitsOverload = cms.PSet(
            initialSeed = cms.untracked.uint32(82)
            ),
                                                       )
    
    process.hltCsc2DRecHitsOverload = cms.EDProducer('CFEBBufferOverloadProducer',
                                                     cscRecHitTag = cms.InputTag("hltCsc2DRecHits"),
                                                     failureRate = cms.untracked.double(0.15),
                                                     doUniformFailure = cms.untracked.bool(True),
                                                     doCFEBFailure = cms.untracked.bool(True),
                                                     )
    
    # change input to cscSegments
    process.hltCscSegments.inputObjects = "hltCsc2DRecHitsOverload"

    # make a new collection of reduced rechits and feed those into the csc segment producer
    process.filteredHltCscSegmentSequence = cms.Sequence(process.hltCsc2DRecHitsOverload + process.hltCscSegments)
    process.HLTMuonLocalRecoSequence.replace(process.hltCscSegments, process.filteredHltCscSegmentSequence)
    
    return process

def rpcAging(process):
    # RPC aging
    
    from L1Trigger.L1IntegratedMuonTrigger.RPCChamberMasker_cff import appendRPCChamberMaskerAtUnpacking2
    process = appendRPCChamberMaskerAtUnpacking2(process,True,[637570221,637602989,637635757,637569201,637634737,637571545,637571677,637637213,637567793,637600561,637633329,637566989,637632525,637566993,637632529,637569037,637634573,637571565,637637101,637571449,637636985,637572317,637637853,637568057,637600825,637633593,637568025,637633561,637567381,637632917,637571805,637637341,637569325,637602093,637634861,637572569,637638105,637571569,637637105,637567065,637632601,637570229,637635765,637567285,637600053,637632821,637572589,637638125,637571157,637636693,637570485,637636021,637571181,637636717,637567069,637632605,637570365,637603133,637635901,637570481,637636017,637572445,637637981,637567005,637632541,637567373,637632909,637568509,637634045,637569425,637634961,637572561,637638097,637568409,637633945,637571281,637636817,637569973,637635509,637567157,637599925,637632693,637571197,637636733,637571321,637636857,637570237,637603005,637635773,637569453,637602221,637634989,637569977,637635513,637568305,637601073,637633841,637570097,637635633,637570197,637635733,637569709,637602477,637635245,637571161,637636697,637572429,637637965,637567669,637600437,637633205,637572045,637637581,637568301,637633837,637567213,637632749,637567801,637600569,637633337,637568053,637600821,637633589,637570109,637602877,637635645,637567965,637633501,637567993,637633529,637637081,637571193,637636729,637567281,637600049,637632817,637567469,637633005,637567953,637633489,637568177,637600945,637633713,637568469,637634005,637567021,637632557,637571317,637636853,637567885,637633421,637572085,637637621,637568213,637633749,637567441,637632977,637569305,637634841,637567341,637632877,637575669,637641205,637567665,637600433,637633201,637567805,637633341,637570233,637635769,637570321,637635857,637567033,637599801,637632569,637568505,637634041,637571453,637636989,637579769,637645305,637572573,637638109,637571277,637636813,637568237,637633773,637571925,637637461,637568089,637633625,637569457,637634993,637572437,637637973,637571961,637637497,637568345,637633881,637572345,637637881,637567189,637632725,637571413,637575620,637608388,637641156,637588100,637620868,637653636,637571816,637604584,637637352,637575462,637608230,637640998,637579430,637612198,637644966,637566980,637599748,637632516,637567242,637600010,637632778,637587818,637620586,637653354,637567722,637600490,637633258,637579784,637612552,637645320,637579658,637612426,637645194,637575242,637608010,637640778,637587976,637620744,637653512,637575466,637608234,637641002,637567524,637600292,637633060,637571658,637604426,637637194,637579882,637612650,637645418,637567462,637600230,637632998,637587556,637620324,637653092,637579942,637612710,637645478,637571210,637603978,637636746,637579748,637612516,637645284,637575338,637608106,637640874,637587976,637620744,637653512,637579910,637612678,637645446,637583944,637616712,637649480,637584106,637616874,637649642,637579782,637612550,637645318,637579592,637612360,637645128])
    
    return process
    

def hltRpcAging(process):
    # RPC aging
    
    from L1Trigger.L1IntegratedMuonTrigger.RPCChamberMasker_cff import appendRPCChamberMaskerAtHLT
    process = appendRPCChamberMaskerAtHLT(process,True,[637570221,637602989,637635757,637569201,637634737,637571545,637571677,637637213,637567793,637600561,637633329,637566989,637632525,637566993,637632529,637569037,637634573,637571565,637637101,637571449,637636985,637572317,637637853,637568057,637600825,637633593,637568025,637633561,637567381,637632917,637571805,637637341,637569325,637602093,637634861,637572569,637638105,637571569,637637105,637567065,637632601,637570229,637635765,637567285,637600053,637632821,637572589,637638125,637571157,637636693,637570485,637636021,637571181,637636717,637567069,637632605,637570365,637603133,637635901,637570481,637636017,637572445,637637981,637567005,637632541,637567373,637632909,637568509,637634045,637569425,637634961,637572561,637638097,637568409,637633945,637571281,637636817,637569973,637635509,637567157,637599925,637632693,637571197,637636733,637571321,637636857,637570237,637603005,637635773,637569453,637602221,637634989,637569977,637635513,637568305,637601073,637633841,637570097,637635633,637570197,637635733,637569709,637602477,637635245,637571161,637636697,637572429,637637965,637567669,637600437,637633205,637572045,637637581,637568301,637633837,637567213,637632749,637567801,637600569,637633337,637568053,637600821,637633589,637570109,637602877,637635645,637567965,637633501,637567993,637633529,637637081,637571193,637636729,637567281,637600049,637632817,637567469,637633005,637567953,637633489,637568177,637600945,637633713,637568469,637634005,637567021,637632557,637571317,637636853,637567885,637633421,637572085,637637621,637568213,637633749,637567441,637632977,637569305,637634841,637567341,637632877,637575669,637641205,637567665,637600433,637633201,637567805,637633341,637570233,637635769,637570321,637635857,637567033,637599801,637632569,637568505,637634041,637571453,637636989,637579769,637645305,637572573,637638109,637571277,637636813,637568237,637633773,637571925,637637461,637568089,637633625,637569457,637634993,637572437,637637973,637571961,637637497,637568345,637633881,637572345,637637881,637567189,637632725,637571413,637575620,637608388,637641156,637588100,637620868,637653636,637571816,637604584,637637352,637575462,637608230,637640998,637579430,637612198,637644966,637566980,637599748,637632516,637567242,637600010,637632778,637587818,637620586,637653354,637567722,637600490,637633258,637579784,637612552,637645320,637579658,637612426,637645194,637575242,637608010,637640778,637587976,637620744,637653512,637575466,637608234,637641002,637567524,637600292,637633060,637571658,637604426,637637194,637579882,637612650,637645418,637567462,637600230,637632998,637587556,637620324,637653092,637579942,637612710,637645478,637571210,637603978,637636746,637579748,637612516,637645284,637575338,637608106,637640874,637587976,637620744,637653512,637579910,637612678,637645446,637583944,637616712,637649480,637584106,637616874,637649642,637579782,637612550,637645318,637579592,637612360,637645128])
    
    return process

def reRunDttf(process):
    from L1Trigger.L1IntegratedMuonTrigger.DTChamberMasker_cff import reRunDttf as DTChamberMasker_reRunDttf
    process = DTChamberMasker_reRunDttf( process ) 
    return process
    
def dtAging(process):
    # DT aging

    from L1Trigger.L1IntegratedMuonTrigger.DTChamberMasker_cff import appendChamberMaskerAtUnpacking
    process = appendChamberMaskerAtUnpacking(process,True,True,[
    # MB4 of top sectors
    "WH-2_ST4_SEC2","WH-2_ST4_SEC3","WH-2_ST4_SEC4","WH-2_ST4_SEC5","WH-2_ST4_SEC6",
    "WH-1_ST4_SEC2","WH-1_ST4_SEC3","WH-1_ST4_SEC4","WH-1_ST4_SEC5","WH-1_ST4_SEC6",
    "WH0_ST4_SEC2","WH0_ST4_SEC3","WH0_ST4_SEC4","WH0_ST4_SEC5","WH0_ST4_SEC6",
    "WH1_ST4_SEC2","WH1_ST4_SEC3","WH1_ST4_SEC4","WH1_ST4_SEC5","WH1_ST4_SEC6",
    "WH2_ST4_SEC2","WH2_ST4_SEC3","WH2_ST4_SEC4","WH2_ST4_SEC5","WH2_ST4_SEC6",
    # MB1 of external wheels
    "WH-2_ST1_SEC1","WH-2_ST1_SEC2","WH-2_ST1_SEC3","WH-2_ST1_SEC4",
    "WH-2_ST1_SEC5","WH-2_ST1_SEC6","WH-2_ST1_SEC7","WH-2_ST1_SEC8",
    "WH-2_ST1_SEC9","WH-2_ST1_SEC10","WH-2_ST1_SEC11","WH-2_ST1_SEC12",
    "WH2_ST1_SEC1","WH2_ST1_SEC2","WH2_ST1_SEC3","WH2_ST1_SEC4",
    "WH2_ST1_SEC5","WH2_ST1_SEC6","WH2_ST1_SEC7","WH2_ST1_SEC8",
    "WH2_ST1_SEC9","WH2_ST1_SEC10","WH2_ST1_SEC11","WH2_ST1_SEC12",
    # 5 MB2s of external wheels
    "WH2_ST2_SEC3","WH2_ST2_SEC6","WH2_ST2_SEC9",
    "WH-2_ST2_SEC2","WH-2_ST2_SEC4",
    # more sparse failures
    "WH-2_ST2_SEC8","WH-1_ST1_SEC1","WH-1_ST2_SEC1","WH-1_ST1_SEC4","WH-1_ST3_SEC7",
    "WH0_ST2_SEC2","WH0_ST3_SEC5","WH0_ST4_SEC12","WH1_ST1_SEC6","WH1_ST1_SEC10","WH1_ST3_SEC3"
    ])
    return process

def hltDtAging(process):
    # DT aging

    from L1Trigger.L1IntegratedMuonTrigger.DTChamberMasker_cff import appendChamberMaskerAtHLT
    process = appendChamberMaskerAtHLT(process,True,True,[
    # MB4 of top sectors
    "WH-2_ST4_SEC2","WH-2_ST4_SEC3","WH-2_ST4_SEC4","WH-2_ST4_SEC5","WH-2_ST4_SEC6",
    "WH-1_ST4_SEC2","WH-1_ST4_SEC3","WH-1_ST4_SEC4","WH-1_ST4_SEC5","WH-1_ST4_SEC6",
    "WH0_ST4_SEC2","WH0_ST4_SEC3","WH0_ST4_SEC4","WH0_ST4_SEC5","WH0_ST4_SEC6",
    "WH1_ST4_SEC2","WH1_ST4_SEC3","WH1_ST4_SEC4","WH1_ST4_SEC5","WH1_ST4_SEC6",
    "WH2_ST4_SEC2","WH2_ST4_SEC3","WH2_ST4_SEC4","WH2_ST4_SEC5","WH2_ST4_SEC6",
    # MB1 of external wheels
    "WH-2_ST1_SEC1","WH-2_ST1_SEC2","WH-2_ST1_SEC3","WH-2_ST1_SEC4",
    "WH-2_ST1_SEC5","WH-2_ST1_SEC6","WH-2_ST1_SEC7","WH-2_ST1_SEC8",
    "WH-2_ST1_SEC9","WH-2_ST1_SEC10","WH-2_ST1_SEC11","WH-2_ST1_SEC12",
    "WH2_ST1_SEC1","WH2_ST1_SEC2","WH2_ST1_SEC3","WH2_ST1_SEC4",
    "WH2_ST1_SEC5","WH2_ST1_SEC6","WH2_ST1_SEC7","WH2_ST1_SEC8",
    "WH2_ST1_SEC9","WH2_ST1_SEC10","WH2_ST1_SEC11","WH2_ST1_SEC12",
    # 5 MB2s of external wheels
    "WH2_ST2_SEC3","WH2_ST2_SEC6","WH2_ST2_SEC9",
    "WH-2_ST2_SEC2","WH-2_ST2_SEC4",
    # more sparse failures
    "WH-2_ST2_SEC8","WH-1_ST1_SEC1","WH-1_ST2_SEC1","WH-1_ST1_SEC4","WH-1_ST3_SEC7",
    "WH0_ST2_SEC2","WH0_ST3_SEC5","WH0_ST4_SEC12","WH1_ST1_SEC6","WH1_ST1_SEC10","WH1_ST3_SEC3"
    ])
    return process

def loadMuonRecHits(process):
    process.load('RecoLocalMuon.RPCRecHit.rpcRecHits_cfi')
    process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
    process.load('RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi')
    return process
    
def runOnlyL2Mu(process):
    process.HLTSchedule = cms.Schedule( cms.Path( process.HLTBeginSequence + process.HLTL2muonrecoSequence + process.HLTL2muonrecoSequenceNoVtx + process.HLTEndSequence))
    process.HLT_L2MuOnly_v1 = cms.Path( process.HLTBeginSequence + process.HLTL2muonrecoSequence + process.HLTL2muonrecoSequenceNoVtx + process.HLTEndSequence)
    process.schedule = cms.Schedule( * [process.HLTriggerFirstPath, process.HLT_L2MuOnly_v1, process.HLTriggerFinalPath, process.HLTAnalyzerEndpath, process.endjob_step, process.RECOSIMoutput_step])
    process.RECOSIMoutput.outputCommands = cms.untracked.vstring( (
            'keep *',
            'drop *_mix_*_*',
            'drop *_simSiPixelDigis_*_*',
            'drop *_simSiStripDigis_*_*',
            'drop *_*_HGCHitsEE_*',
            'drop *_*_TrackerHitsPixel*_*',
            'drop *_*_EcalHitsEB_*',
            'drop *_*_EcalHitsEE_*',
            'drop *_simHcal*_*_*',
            'drop *_simEcal**_*_*',
            'drop *_*_HcalHits_*',
            'drop *_*_BSCHits_*',
            'drop PCaloHits_*_*_*',
            'drop *_hltGctDigis_*_*',
            'drop *_*GenJets_*_*',
            'drop *_simGctDigis_*_*',
            'drop *_hltScalersRawToDigi_*_*',
            'drop *_*_TrackerHits*_*',
            'drop *_genMet*_*_*',
            'drop *_*_TotemHits*_*',
            'drop *_*_FastTimerHits_*',
            'drop *_*_BHMHits_*',
            'drop *_*_FP420SI_*',
            'drop *_*_PLTHits_*',
            'drop *_rawDataCollector_*_*',
            'drop *_randomEngineStateProducer_*_*',
            )
    )
    return process

def fullScopeDetectors(process):
    # CSC
    # case 0  :: all detectors in
    # case 1  :: ME1/1 switched off
    # case 2  :: ME2/1 switched off
    # case 3  :: ME3/1 switched off
    # case 4  :: ME4/1 switched off
    # default :: all detectors in
    
    # RPC
    # case 0  :: NO RPC Upgrade
    # case 1  :: RE3/1 switched off
    # case 2  :: RE4/1 switched off
    # case 3  :: RPC Upgrade switched on
    # default :: all detectors in
    
    #GEM
    # case 0  :: all detectors off
    # case 1  :: GE1/1 switched on
    # case 2  :: GE2/1 switched on
    # case 3  :: all detectors in
    # default :: all detectors in
    process = loadMuonRecHits(process)
    process.csc2DRecHits.stationToUse = cms.untracked.int32(0)
    process.rpcRecHits.recAlgoConfig.stationToUse = cms.untracked.int32(3)
    process.gemRecHits.recAlgoConfig.stationToUse = cms.untracked.int32(3)
    return process

def descope235MCHFDetectors(process):
    #235 MCHF: RE3/1 + RE4/1 switched off
    process = loadMuonRecHits(process)
    process.csc2DRecHits.stationToUse = cms.untracked.int32(0)
    process.rpcRecHits.recAlgoConfig.stationToUse = cms.untracked.int32(0)
    process.gemRecHits.recAlgoConfig.stationToUse = cms.untracked.int32(3)
    return process

def descope200MCHFDetectors(process):
    #200 MCHF: GE2/1 + RE3/1 + RE4/1 switched off
    process = loadMuonRecHits(process)
    process.csc2DRecHits.stationToUse = cms.untracked.int32(0)
    process.rpcRecHits.recAlgoConfig.stationToUse = cms.untracked.int32(0)
    process.gemRecHits.recAlgoConfig.stationToUse = cms.untracked.int32(1)
    return process

def applyAgingToL2MuNoDT(process):

    if hasattr(process,'HLTMuonLocalRecoSequence') :
        # RPC
        # The aging for the RPC system is applied to the digis
        # but the packing/unpacking does not work
        # so we need to pass the simMuonRPCDigis to the hltRpcRecHits
        process = hltRpcAging(process)
        
        # CSC
        # The aging is applied to the rechits
        # so we need to pass the reduced collection (csc2DRecHitsOverload) to the hltCSCSegments
        # this is already done in the aging functions above, so we only need to change the inputs
        # to the L2Mu reconstruction
        process = hltCscAging(process)

    return process

def applyAgingToL2Mu(process):

    if hasattr(process,'HLTMuonLocalRecoSequence') :
        # RPC
        # The aging for the RPC system is applied to the digis
        # but the packing/unpacking does not work
        # so we need to pass the simMuonRPCDigis to the hltRpcRecHits
        process = hltRpcAging(process)
        
        # CSC
        # The aging is applied to the rechits
        # so we need to pass the reduced collection (csc2DRecHitsOverload) to the hltCSCSegments
        # this is already done in the aging functions above, so we only need to change the inputs
        # to the L2Mu reconstruction
        process = hltCscAging(process)

        # DT
        # The aging is applied to the digis
        # We assume the packing/unpacking works, so nothing needs to be done
        process = hltDtAging(process)

    return process

def fullScope(process):
    process = fullScopeDetectors(process)
    process = runOnlyL2Mu(process)
    return process

def descope235MCHF(process):
    process = descope235MCHFDetectors(process)
    process = runOnlyL2Mu(process)
    return process

def descope200MCHF(process):
    process = descope200MCHFDetectors(process)
    process = runOnlyL2Mu(process)
    return process



def fullScopeAging(process):
    process = fullScopeDetectors(process)
    process = runOnlyL2Mu(process)
    process = applyAgingToL2MuNoDT(process)
    return process
 
def descope235MCHFaging(process):
    process = descope235MCHFDetectors(process)
    process = runOnlyL2Mu(process)
    process = applyAgingToL2MuNoDT(process)
    return process

def descope200MCHFaging(process):
    process = descope200MCHFDetectors(process)
    process = runOnlyL2Mu(process)
    process = applyAgingToL2Mu(process)
    return process
