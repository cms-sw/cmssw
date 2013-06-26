import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonsTriggerPlots")

### standard includes
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

### source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'root:://pcmssd12.cern.ch//data/gpetrucc/7TeV/hlt/MC_vecchio.root' ## doesn't actually have HLT debug inside :-(
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

## Apply other trigger skims (to handle with prescaled triggers)
from HLTrigger.HLTfilters.hltHighLevelDev_cfi import hltHighLevelDev
process.bit40 = hltHighLevelDev.clone(HLTPaths = ['HLT_MinBiasBSC'], HLTPathsPrescales = [1])
process.halo  = hltHighLevelDev.clone(HLTPaths = ['HLT_L1Tech_BSC_halo'], HLTPathsPrescales = [1])

## Quality filter on the muons
process.patMuons = cms.EDFilter("PATMuonSelector", src = cms.InputTag("patMuonsWithTrigger"), cut = cms.string("isGlobalMuon && muonID('GlobalMuonPromptTight')"))
## At least one good muon
process.muonFilter       = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("patMuons"), minNumber = cms.uint32(1))
## No more than one (maxNumber didn't work for me, I don't know why)
process.diMuonFilter     = process.muonFilter.clone(minNumber = 2) 
process.singleMuonFilter = cms.Sequence(~process.diMuonFilter)

process.patMuonTree = cms.EDAnalyzer("ProbeTreeProducer",
    src = cms.InputTag("patMuons"),
    cut = cms.string(""), # you might want a quality cut here
    variables = cms.PSet(
        ## Variables of the muon
        pt  = cms.string("pt"),
        p   = cms.string("p"),
        eta = cms.string("eta"),
        phi = cms.string("phi"),
        ## Variables of the L1, taken from AOD info
        l1q  = cms.string("userInt('muonL1Info:quality')"),
        l1dr = cms.string("userFloat('muonL1Info:deltaR')"),
        ## HLT DEBUG info. They'll be 0 if fail, or N>=1 if succeed
        d_PropToM2       = cms.string("userInt('matchDebug:propagatesToM2')"), # 1 if it propagates to M2
        d_L1Any          = cms.string("userInt('matchDebug:hasL1Particle')"),  # number of L1 candidates in cone
        d_L1SingleMuOpen = cms.string("userInt('matchDebug:hasL1Filtered')"),  # number of L1 candidates passing filter
        d_L2Seed         = cms.string("userInt('matchDebug:hasL2Seed')"),      # as above but giving L2 seed
        d_L2MuOpen       = cms.string("userInt('matchDebug:hasL2Muon')"),      # ... I think you can guess the rest
        d_L2Mu3          = cms.string("userInt('matchDebug:hasL2MuonFiltered')"),
        d_L3Seed         = cms.string("userInt('matchDebug:hasL3Seed')"),
        d_L3Track        = cms.string("userInt('matchDebug:hasL3Track')"),
        d_L3MuOpen       = cms.string("userInt('matchDebug:hasL3Muon')"),
        d_L3Mu3          = cms.string("userInt('matchDebug:hasL3MuonFiltered')"),
    ),
    flags = cms.PSet(
        # On MC, true if matched to a prompt muon
        mc = cms.string("genParticleRef.isNonnull"),
        # TRIGGER INFO from AOD matching
        PropToM2       = cms.string("!triggerObjectMatchesByFilter('propagatedToM2').empty()"),  # Can prop. to M2
        L1Any          = cms.string("userCand('muonL1Info').isNonnull"),                         # At least one L1 candidate
        L1Pt3          = cms.string("userCand('muonL1Info').isNonnull && userCand('muonL1Info').pt >= 3"),  # Example of selected L1 candidate: req. pt >= 3
        L1SingleMuOpen = cms.string("!triggerObjectMatchesByFilter('hltL1MuOpenL1Filtered0').empty()"),     # Passes last filter of L1SingleMuOpen
        L2Mu0          = cms.string("!triggerObjectMatchesByFilter('hltL2Mu0L2Filtered0').empty()"),        # Passes last filter of ...
        L2Mu3          = cms.string("!triggerObjectMatchesByFilter('hltSingleMu3L2Filtered3').empty()"),    # ...
        L2Mu9          = cms.string("!triggerObjectMatchesByFilter('hltL2Mu9L2Filtered9').empty()"),
        Mu3            = cms.string("!triggerObjectMatchesByFilter('hltSingleMu3L3Filtered3').empty()"),
        Mu5            = cms.string("!triggerObjectMatchesByFilter('hltSingleMu5L3Filtered5').empty()"),
        Mu9            = cms.string("!triggerObjectMatchesByFilter('hltSingleMu9L3Filtered9').empty()"),
    ),
)

process.p2 = cms.Path(
    process.bit40 + #~process.halo +
    process.patMuons            +
    process.muonFilter          +
    process.singleMuonFilter    +
    process.patMuonTree         
)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.TFileService = cms.Service("TFileService", fileName = cms.string("plots.hltdebug.root"))

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange(
#        '132440:85-132440:138',
#        '132440:141-132440:401',
#        '132596:382-132596:383',
#        '132596:447-132596:453',
#        '132598:80-132598:82',
#        '132598:174-132598:188',
#        '132599:1-132599:379',
#        '132599:381-132599:538',
#        '132601:1-132601:207',
#        '132601:209-132601:259',
#        '132601:261-132601:1131',
#        '132602:1-132602:83',
#        '132605:1-132605:444',
#        '132605:446-132605:622',
#        '132605:624-132605:829',
#        '132605:831-132605:968',
#        '132606:1-132606:37',
#        '132656:1-132656:140',
#        '132658:1-132658:177',
#        '132659:1-132659:84',
#        '132661:1-132661:130',
#        '132662:1-132662:130',
#        '132662:132-132662:217',
#        '132716:220-132716:591',
#        '132716:593-132716:640',
#)
#
