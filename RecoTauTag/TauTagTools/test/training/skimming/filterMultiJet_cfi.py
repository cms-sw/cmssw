import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.RecoTauCommonJetSelections_cfi as common

filterConfig = cms.PSet(
    name = cms.string("MultiJet"),
    hltPaths = cms.vstring("HLT_Jet15U", "HLT_Jet30U", "HLT_Jet50U"),
    # Flag to specify whether we want to use (unbiased) jets that are matched to our
    useUnbiasedHLTMatchedJets = cms.bool(True)
)
useUnbiasedHLTMatchedJets = True

# Quickly make sure there are at least 2 jets in the event
selectedRecoJetsForQCDFilter = cms.EDFilter(
    "CandViewRefSelector",
    src = common.jet_collection,
    cut = common.kinematic_selection,
    filter = cms.bool(True)
)

# We need at least two jets (to match to the trigger)
atLeastTwoRecoJetsForQCDFilter = cms.EDFilter(
    'CandViewCountFilter',
    src = cms.InputTag("selectedRecoJetsForQCDFilter"),
    minNumber = cms.uint32(2),
)

selectEnrichedEvents = cms.Sequence(
    selectedRecoJetsForQCDFilter +
    atLeastTwoRecoJetsForQCDFilter
)

# dbs search --noheader --query="find file where primds=RelValQCD_FlatPt* and release=$CMSSW_VERSION and tier=GEN-SIM-RECO"  | sed "s|.*|\"&\",|"
filterConfig.testFiles = cms.vstring([
    "/store/relval/CMSSW_3_11_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_311_V1_64bit-v1/0089/C07F1687-1535-E011-A7A0-001A92810ADE.root",
    "/store/relval/CMSSW_3_11_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_311_V1_64bit-v1/0089/B8D9658A-1635-E011-9498-0018F3D09688.root",
    "/store/relval/CMSSW_3_11_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_311_V1_64bit-v1/0089/AA6DE9FC-0D35-E011-A35A-0018F3D095F2.root",
    "/store/relval/CMSSW_3_11_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_311_V1_64bit-v1/0089/3C542B7F-0D35-E011-B0B7-0018F3D095EA.root",
    "/store/relval/CMSSW_3_11_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_311_V1_64bit-v1/0089/2ADFF002-1935-E011-944A-002618943858.root",
    "/store/relval/CMSSW_3_11_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_311_V1_64bit-v1/0089/2A65AB6A-0C35-E011-B6E6-002618943954.root",
])
