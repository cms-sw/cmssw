import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTHiggsOfflineAnalysis")

#process.load("DQMOffline.RecoB.dqmAnalyzer_cff")
process.load("HLTriggerOffline.Higgs.dqmAnalyzer_cff")
process.load("HLTriggerOffline.Higgs.HiggsValidation_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")


##############################################################################
##### Templates to change parameters in hltMuonValidator #####################
# process.hltMuonValidator.hltPathsToCheck = ["HLT_IsoMu3"]
# process.hltMuonValidator.genMuonCut = "abs(mother.pdgId) == 24"
# process.hltMuonValidator.recMuonCut = "isGlobalMuon && eta < 1.2"
##############################################################################

hltProcessName = "HLT"
process.hltHiggsValidator.hltProcessName = hltProcessName
#process.hltHiggsValidator.HWW.hltPathsToCheck = cms.vstring(
#		"HLT_Photon26",
		#		"HLT_Mu30_eta2p1_v",
		#		"HLT_IsoMu24_eta2p1_v",
		#"HLT_Ele27_WP80_v",
#		)

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string(autoCond['startup'])

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
            #'file:/afs/cern.ch/user/d/duarte/scratch0/step2_RAW2DIGI_RECO.root',
             #'/store/relval/CMSSW_6_2_0/RelValQCD_Pt_600_800/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/B0E46DF7-5CEC-E211-BAAD-0025901D5C7E.root',
            #'/store/relval/CMSSW_5_2_0/RelValZMM/GEN-SIM-RECO/START52_V4A-v1/0248/BE2AD2B0-1569-E111-B555-003048678FF4.root',
            #'/store/relval/CMSSW_6_2_0/RelValMinBias/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/E25D61DF-53EC-E211-833B-002481E736D2.root',

            #'/store/relval/CMSSW_6_2_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/F640D45E-47EC-E211-8B37-00237DDC5C16.root',

#'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_6_2_0/src/Configuration/Generator/test/step2_RAW2DIGI_RECO.root'
            'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_7_2_0_pre1/src/RECO/0090828A-AC71-E311-A488-7845C4FC36D7.root'
            #'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_6_2_0/src/AODSIM/8A160446-E270-E311-B293-00266CFAC810.root',
            #'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_6_2_0/src/AODSIM/7691F031-1C70-E311-8935-7845C4F9321B.root'
#'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_6_2_0/src/AODSIM/703841B6-C271-E311-9BE1-848F69FD501B.root',

#'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_6_2_0/src/AODSIM/CE4C9570-0379-E311-8A51-7845C4FC3611.root',

#'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_6_2_0/src/AODSIM/A2CE01C5-B46F-E311-8067-00A0D1EE9424.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
            #'file:/afs/cern.ch/user/d/duarte/scratch0/H130GGgluonfusion_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root',
           'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_7_2_0_pre1/src/GEN-SIM-RAW/0090828A-AC71-E311-A488-7845C4FC36D7.root'
    )
)


process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000
process.MessageLogger.destinations += ['HiggsValidationMessages']
process.MessageLogger.categories   += ['HiggsValidation']
process.MessageLogger.debugModules += ['*']#HLTHiggsValidator','HLTHiggsSubAnalysis','HLTHiggsPlotter']
process.MessageLogger.HiggsValidationMessages = cms.untracked.PSet(
    threshold       = cms.untracked.string('DEBUG'),
    default         = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    HiggsValidation = cms.untracked.PSet(limit = cms.untracked.int32(1000))
    )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_HLTHiggsOfflineAnalysis'),
    fileName = cms.untracked.string('hltHiggsValidator.root')
)


process.analyzerpath = cms.Path(
    process.prebTagSequence *
    process.hltHiggsValidator *
    process.MEtoEDMConverter
)


process.outpath = cms.EndPath(process.out)
