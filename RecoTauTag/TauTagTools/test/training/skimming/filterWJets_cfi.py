import FWCore.ParameterSet.Config as cms

'''

Code to select enriched sample of W+jets

Authors: Christian Veelken, Evan Friis (UC Davis)

'''

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *

# Turn off MC matching
patMuons.addGenMatch = False
patMuons.embedGenMatch = False
patMuons.genParticleMatch = ''
makePatMuons.remove(muonMatch)

transverse_mass_str = 'sqrt(abs(daughter(0).et*daughter(1).et -'\
        'daughter(0).px*daughter(1).px -'\
        'daughter(0).py*daughter(1).py))'

# Have to put all this in a pset so it gets imported.
filterConfig = cms.PSet(
    name = cms.string("WJets"),
    hltPaths = cms.vstring(
        'HLT_Mu9', 'HLT_IsoMu9', 'HLT_Mu11',
        'HLT_IsoMu13_v3', 'HLT_IsoMu13_v4', 'HLT_Mu15_v1'),
    #hltPaths = cms.vstring('HLT_Mu9'),
    # Flag to specify whether we want to use (unbiased) jets that are matched to our
    # trigger.  Not for this case.
    useUnbiasedHLTMatchedJets = cms.bool(False),
)

#--------------------------------------------------------------------------------
#
# define selection of "loose" Muons
#
patGlobalMuons = cms.EDFilter(
    "PATMuonSelector",
    src = cms.InputTag('patMuons'),
    cut = cms.string("isGlobalMuon"),
    filter = cms.bool(False)
)

# Veto events w/ two global muons
diMuonVeto = cms.EDFilter(
    "PATCandViewCountFilter",
    src = cms.InputTag('patGlobalMuons'),
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(1)
)

#
# define selection of "tight" Muons
#
selectedMuons = cms.EDFilter(
    "PATMuonSelector",
    src = cms.InputTag('patMuons'),
    cut = cms.string("isGlobalMuon & pt > 15. & abs(eta) < 2.1"
                     "& (trackIso() + ecalIso() + hcalIso()) < 0.1*pt"),
    filter = cms.bool(True)
)

selectedRecoJetsForBgFilter = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("ak5PFJets"),
    cut = cms.string("abs(eta) < 2.5 & pt > 10"),
    filter = cms.bool(True)
)

muonMETPairs = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string(transverse_mass_str + ' > 40'),
    decay = cms.string("selectedMuons pfMet")
)

# check the MT cut passes
muonMETCut = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("muonMETPairs"),
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(1)
)

muonJetPairs = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string(
        "deltaR(daughter(0).eta, daughter(0).phi,"
        "daughter(1).eta, daughter(1).phi) > 0.7"),
    decay = cms.string("selectedMuons selectedRecoJetsForBgFilter")
)

muonJetPairFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("muonJetPairs"),
    minNumber = cms.uint32(1)
)

selectEnrichedEvents = cms.Sequence(
    makePatMuons +
    patGlobalMuons +
    diMuonVeto +
    selectedMuons +
    selectedRecoJetsForBgFilter +
    muonMETPairs +
    muonMETCut +
    muonJetPairs +
    muonJetPairFilter
)

#  dbs search --query="find file where primds=RelValWM and release=CMSSW_3_11_1 and tier=GEN-SIM-RECO"
filterConfig.testFiles = cms.vstring([
    "/store/relval/CMSSW_4_2_0_pre5/RelValWM/GEN-SIM-RECO/START42_V3-v1/0012/E21A1F98-F03C-E011-89A4-002618943896.root",
    "/store/relval/CMSSW_4_2_0_pre5/RelValWM/GEN-SIM-RECO/START42_V3-v1/0009/C0EC55BB-4B3C-E011-856D-00304867BECC.root",
    "/store/relval/CMSSW_4_2_0_pre5/RelValWM/GEN-SIM-RECO/START42_V3-v1/0009/927C3919-8A3C-E011-B22B-0018F3D09704.root",
    "/store/relval/CMSSW_4_2_0_pre5/RelValWM/GEN-SIM-RECO/START42_V3-v1/0009/589B4414-593C-E011-AB4D-0018F3D09600.root",
])
