import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTGenValSource")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Core.DQMStore_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.FwkReport.reportEvery = 100


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/03a15a72-f9e3-4488-b0ca-3c9367d94e03.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/262f6750-5888-47c8-9ec1-89a6e19232a7.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/2ffe551c-6d4a-4852-9730-a2bb9c7aa1d3.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/37b3f169-2152-4485-80d7-cf415847712d.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/56ac8b93-2d72-4e30-b7fb-a82d70ae02b9.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/614a0f55-2502-4d27-90e3-b6cada9c25c5.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/678f6818-986e-4486-8c56-efbe080fca6a.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/6883cf9f-db51-4b54-9a87-2f35aa87c140.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/6b321170-7d05-4eab-84e5-a6b3bd244b41.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/7db7cf3d-01d5-4068-ae1e-5212a9acd2e2.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/8c3487e5-dc7c-45e6-955c-40972e7585dc.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/8ca081bf-b616-465a-a3a6-313a1f6323ed.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/8fe32e18-9e98-4cd7-96a8-9e8c2b8624f6.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/acbf7935-4c19-4618-888e-2bfb751e07a2.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/bd0c5382-f9ab-4299-815a-2c7d6c902fdd.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/c0c69b4c-c7f7-4023-8a21-1e0002672b19.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/ca441daf-c921-4276-be9b-20e5dc8d30c8.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/cb9cb297-5e11-4ae9-8b9f-be3b81043764.root",
        "/store/relval/CMSSW_14_1_0_pre2/RelValSMS-T1tttt_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v3_STD_2026D98_noPU-v6/2580000/d906f6d6-fdb7-4a10-8c17-0f31b00d6140.root"
    )
)

ptBins = cms.vdouble(0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 , 100, 105, 110, 115 , 120, 125, 130, 135, 140, 145, 150)
ptBinsHT = cms.vdouble(0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300)
ptBinsMET = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300)
ptBinsJet = cms.vdouble(0, 100, 200, 300, 350, 375, 400, 425, 450, 475, 500, 550, 600, 700, 800, 900, 1000)
etaBins = cms.vdouble(-4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4)

etaCut = cms.PSet(
    rangeVar=cms.string("eta"),
    allowedRanges=cms.vstring("-2.4:2.4")
)
ptCut = cms.PSet(
    rangeVar=cms.string("pt"),
    allowedRanges=cms.vstring("40:9999")
)

hltProcessName = cms.string("HLT")
doOnlyLastFilter = cms.bool(True)

# Photons pass electron paths too
# Hence the same sample can be used to validate both electrons and photons
process.HLTGenValSourceELE = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("ele"),
    hltProcessName = hltProcessName,
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleEle23_12_Iso_L1Seeded",
        "HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded",
        "HLT_DoubleEle25_CaloIdL_PMS2_Unseeded",
        "HLT_Ele115_NonIso_L1Seeded",
        "HLT_Ele26_WP70_L1Seeded",
        "HLT_Ele26_WP70_Unseeded",
        "HLT_Ele32_WPTight_L1Seeded",
        "HLT_Ele32_WPTight_Unseeded",
        "HLT_Diphoton30_23_IsoCaloId_L1Seeded",
        "HLT_Diphoton30_23_IsoCaloId_Unseeded",
        "HLT_Photon108EB_TightID_TightIso_L1Seeded",
        "HLT_Photon108EB_TightID_TightIso_Unseeded",
        "HLT_Photon187_L1Seeded",
        "HLT_Photon187_Unseeded"
    ),
    doOnlyLastFilter = doOnlyLastFilter,
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBins,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins
        ),
    ),
)

# To enable or not: TBD
#process.HLTGenValSourcePHO = cms.EDProducer('HLTGenValSource',
#    # these are the only one the user needs to specify
#    objType = cms.string("pho"),
#    hltProcessName = hltProcessName,
#    hltPathsToCheck = cms.vstring(
#        "HLT_Diphoton30_23_IsoCaloId_L1Seeded",
#        "HLT_Diphoton30_23_IsoCaloId_Unseeded",
#        "HLT_Photon108EB_TightID_TightIso_L1Seeded",
#        "HLT_Photon108EB_TightID_TightIso_Unseeded",
#        "HLT_Photon187_L1Seeded",
#        "HLT_Photon187_Unseeded",
#    ),
#    doOnlyLastFilter = doOnlyLastFilter,
#    histConfigs = cms.VPSet(
#        cms.PSet(
#            vsVar = cms.string("pt"),
#            binLowEdges = ptBins,
#            rangeCuts = cms.VPSet(etaCut)
#        ),
#        cms.PSet(
#            vsVar = cms.string("eta"),
#            binLowEdges = etaBins,
#        ),
#    ),
#)

process.HLTGenValSourceMU = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("mu"),
    hltProcessName = hltProcessName,
    hltPathsToCheck = cms.vstring(
        "HLT_IsoMu24_FromL1TkMuon",
        "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon",
        "HLT_Mu37_Mu27_FromL1TkMuon",
        "HLT_Mu50_FromL1TkMuon",
        "HLT_TriMu_10_5_5_DZ_FromL1TkMuon"
    ),
    doOnlyLastFilter = doOnlyLastFilter,
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBins,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins
        ),
    ),
)

process.HLTGenValSourceTAU = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("tau"),
    hltProcessName = hltProcessName,
    hltPathsToCheck = cms.vstring(
        "HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1",
        "HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1"
    ),
    doOnlyLastFilter = doOnlyLastFilter,
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBins,
            rangeCuts = cms.VPSet(etaCut)
        ),
        cms.PSet(
            vsVar = cms.string("eta"),
            binLowEdges = etaBins
        ),
    ),
)

process.HLTGenValSourceJET = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("AK4jet"),
    hltProcessName = hltProcessName,
    hltPathsToCheck = cms.vstring(
        "HLT_AK4PFPuppiJet520"
    ),
    inputCollections = cms.PSet(
        ak4GenJets = cms.InputTag("ak4GenJetsNoNu"),
    ),
    doOnlyLastFilter = doOnlyLastFilter,
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsJet
        ),
    ),
)

process.HLTGenValSourceHT = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("AK4HT"),
    hltProcessName = hltProcessName,
    hltPathsToCheck = cms.vstring(
        "HLT_PFPuppiHT1070"
    ),
    inputCollections = cms.PSet(
        ak4GenJets = cms.InputTag("ak4GenJetsNoNu"),
    ),
    doOnlyLastFilter = doOnlyLastFilter,
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsHT
        ),
    ),
)

process.HLTGenValSourceMET = cms.EDProducer('HLTGenValSource',
    # these are the only one the user needs to specify
    objType = cms.string("MET"),
    hltProcessName = hltProcessName,
    hltPathsToCheck = cms.vstring(
        "HLT_PFPuppiMETTypeOne140_PFPuppiMHT140"
    ),
    doOnlyLastFilter = doOnlyLastFilter,
    histConfigs = cms.VPSet(
        cms.PSet(
            vsVar = cms.string("pt"),
            binLowEdges = ptBinsMET
        ),
    ),
)



process.p = cms.Path(
    process.HLTGenValSourceELE *
    #process.HLTGenValSourcePHO *
    process.HLTGenValSourceMU *
    process.HLTGenValSourceTAU *
    process.HLTGenValSourceJET *
    process.HLTGenValSourceHT *
    process.HLTGenValSourceMET
)

# the harvester
process.harvester = DQMEDHarvester("HLTGenValClient",
    outputFileName = cms.untracked.string('Phase2_HLT_validation_output.root'),
    subDirs        = cms.untracked.vstring("HLTGenVal")
)

process.outpath = cms.EndPath(process.harvester)
