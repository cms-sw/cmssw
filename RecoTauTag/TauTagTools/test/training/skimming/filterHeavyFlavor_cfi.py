import FWCore.ParameterSet.Config as cms

'''

Code to select enriched sample of mu-enriched QCD

Authors: Christian Veelken, Evan Friis (UC Davis)

'''

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *

# Based on W+jets code
import filterWJets_cfi as wjets

filterConfig = cms.PSet(
    hltPaths = wjets.filterConfig.hltPaths,
    # Flag to specify whether we want to use (unbiased) jets that are matched to our
    # trigger.  It's okay if our jets have muons in them.
    useUnbiasedHLTMatchedJets = cms.bool(True),
    name = cms.string("heavyFlavor"),
)

# define selection of "loose" Muons
#
patGlobalMuons = wjets.patGlobalMuons
diMuonVeto = wjets.diMuonVeto
# Loosely invert isolation
selectedMuons = wjets.selectedMuons.clone(
    cut = cms.string("isGlobalMuon & pt > 15. & abs(eta) < 2.1"
                     "& (trackIso() + ecalIso() + hcalIso()) > 0.1*pt"
                     "& (trackIso() + ecalIso() + hcalIso()) < 0.3*pt")
)

selectedRecoJetsForBgFilter = wjets.selectedRecoJetsForBgFilter
# Invert MT cut
muonMETPairs = wjets.muonMETPairs.clone(
    cut = cms.string(wjets.transverse_mass_str + ' < 40')
)
muonMETCut = wjets.muonMETCut

muonJetPairs = wjets.muonJetPairs
muonJetPairFilter = wjets.muonJetPairFilter

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
#  dbs search --noheader --query="find file where primds=RelValJpsiMM and release=$CMSSW_VERSION and tier=GEN-SIM-RECO"  | sed "s|.*|\"&\",|"
filterConfig.testFiles = cms.vstring([
    "/store/relval/CMSSW_3_11_1/RelValJpsiMM/GEN-SIM-RECO/START311_V1_64bit-v1/0090/BECF23AB-CC35-E011-8575-001A92971AA4.root",
    "/store/relval/CMSSW_3_11_1/RelValJpsiMM/GEN-SIM-RECO/START311_V1_64bit-v1/0089/F89297CF-F634-E011-B188-001A92810A9E.root",
    "/store/relval/CMSSW_3_11_1/RelValJpsiMM/GEN-SIM-RECO/START311_V1_64bit-v1/0089/A481AA33-EF34-E011-8D11-0026189438EA.root",
    "/store/relval/CMSSW_3_11_1/RelValJpsiMM/GEN-SIM-RECO/START311_V1_64bit-v1/0089/98466043-E934-E011-B4C1-0018F3D096A0.root",
    "/store/relval/CMSSW_3_11_1/RelValJpsiMM/GEN-SIM-RECO/START311_V1_64bit-v1/0089/36AD4A45-F334-E011-B9DE-001A92810AC4.root",
    "/store/relval/CMSSW_3_11_1/RelValJpsiMM/GEN-SIM-RECO/START311_V1_64bit-v1/0089/32B18641-ED34-E011-A5F7-001A92810AC8.root",
])
