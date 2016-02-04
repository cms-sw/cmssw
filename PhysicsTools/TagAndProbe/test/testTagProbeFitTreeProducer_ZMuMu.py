import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

######### EXAMPLE CFG 
###  A simple test of runnning T&P on Zmumu to determine muon isolation and identification efficiencies
###  More a showcase of the tool than an actual physics example

process.load('FWCore.MessageService.MessageLogger_cfi')
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/D68EFA42-9FE2-DE11-9CEC-002618943856.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/D68C1E7D-8EE2-DE11-A6DE-002618943843.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/D4D52FFE-8CE2-DE11-B614-0026189438F4.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/C4FDD49F-A6E2-DE11-8F4F-001A9281170A.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/C47F4CB3-A3E2-DE11-B364-001731AF6933.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/BE0507A9-A5E2-DE11-97E0-001731AF669D.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/B6112976-8FE2-DE11-BB92-0026189437E8.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/B24A3001-8CE2-DE11-9D8B-0026189438BC.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/B0493230-A4E2-DE11-88C0-001731AF66BD.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/AE2254FE-8CE2-DE11-A10D-002618943961.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/A8C333FB-8DE2-DE11-91F3-003048678D86.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/A4A2E76C-92E2-DE11-ADA5-0026189437E8.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/A4286326-A5E2-DE11-BA9C-001A9281170A.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/A2B8E3C6-9BE2-DE11-97BD-0018F3D09648.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/9EE00419-AAE2-DE11-853D-001731AF6A49.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/90ECED25-A7E2-DE11-ABB3-0018F3D0970A.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/8AC633FA-90E2-DE11-9558-002618943877.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/8A3213FA-90E2-DE11-8C98-002354EF3BD2.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/881A239C-A8E2-DE11-9FA3-00173199E924.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/78A721F9-8EE2-DE11-8ABD-002618943879.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/6823076D-92E2-DE11-B4FA-002618FDA20E.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/66C7B8F3-8FE2-DE11-9760-0018F3D096BC.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/66185789-8DE2-DE11-8953-002618FDA250.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/609FD873-91E2-DE11-93F5-002618943916.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/4ED2B082-ACE2-DE11-9D98-001A92810A92.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/348CC0B5-A1E2-DE11-A714-001A92971B80.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/2AF34C6F-90E2-DE11-9327-002354EF3BD2.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/2A5A24BC-A0E2-DE11-A367-001A9281170A.root',
	'/store/relval/CMSSW_3_4_0_pre7/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_3XY_V14_LowLumiPileUp-v1/0004/1843924A-09E3-DE11-B0D9-002618943919.root',
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )    

## Tags. In a real analysis we should require that the tag muon fires the trigger, 
##       that's easy with PAT muons but not RECO/AOD ones, so we won't do it here
##       (the J/Psi example shows it)
process.tagMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isGlobalMuon && pt > 20 && abs(eta) < 2"), 
)
## Probes. Now we just use Tracker Muons as probes
process.probeMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isTrackerMuon && pt > 10"), 
)

## Here we show how to define passing probes with a selector
## although for this case a string cut in the TagProbeFitTreeProducer would be enough
process.probesPassingCal = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string(process.probeMuons.cut.value() + " && caloCompatibility > 0.6"),
)

## Here we show how to use a module to compute an external variable
process.drToNearestJet = cms.EDProducer("DeltaRNearestJetComputer",
    probes = cms.InputTag("muons"),
       # ^^--- NOTA BENE: if probes are defined by ref, as in this case, 
       #       this must be the full collection, not the subset by refs.
    objects = cms.InputTag("ak5CaloJets"),
    objectSelection = cms.InputTag("et > 20 && abs(eta) < 3 && n60 > 3 && (.05 < emEnergyFraction < .95)"),
)

## Combine Tags and Probes into Z candidates, applying a mass cut
process.tpPairs = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("tagMuons@+ probeMuons@-"), # charge coniugate states are implied
    cut   = cms.string("40 < mass < 200"),
)

## Match muons to MC
process.muMcMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("muons"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("genParticles")
)

## Make the tree
process.muonEffs = cms.EDAnalyzer("TagProbeFitTreeProducer",
    # pairs
    tagProbePairs = cms.InputTag("tpPairs"),
    arbitration   = cms.string("OneProbe"),
    # variables to use
    variables = cms.PSet(
        ## methods of reco::Candidate
        eta = cms.string("eta"),
        pt  = cms.string("pt"),
        ## a method of the reco::Muon object (thanks to the 3.4.X StringParser)
        nsegm = cms.string("numberOfMatches"), 
        ## this one is an external variable
        drj = cms.InputTag("drToNearestJet"),
    ),
    # choice of what defines a 'passing' probe
    flags = cms.PSet(
        ## one defined by an external collection of passing probes
        passingCal = cms.InputTag("probesPassingCal"), 
        ## two defined by simple string cuts
        passingGlb = cms.string("isGlobalMuon"),
        passingIso = cms.string("(isolationR03.hadEt+isolationR03.emEt+isolationR03.sumPt) < 0.1 * pt"),
    ),
    # mc-truth info
    isMC = cms.bool(True),
    motherPdgId = cms.vint32(22,23),
    makeMCUnbiasTree = cms.bool(True),
    checkMotherInUnbiasEff = cms.bool(True),
    tagMatches = cms.InputTag("muMcMatch"),
    probeMatches  = cms.InputTag("muMcMatch"),
    allProbes     = cms.InputTag("probeMuons"),
)
##    ____       _   _     
##   |  _ \ __ _| |_| |__  
##   | |_) / _` | __| '_ \ 
##   |  __/ (_| | |_| | | |
##   |_|   \__,_|\__|_| |_|
##                         
process.tagAndProbe = cms.Path( 
    (process.tagMuons + process.probeMuons) *   # 'A*B' means 'B needs output of A'; 
    (process.probesPassingCal +                 # 'A+B' means 'if you want you can re-arrange the order'
     process.drToNearestJet   +
     process.tpPairs +
     process.muMcMatch) *
    process.muonEffs
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("testTagProbeFitTreeProducer_ZMuMu.root"))




