import FWCore.ParameterSet.Config as cms

process = cms.Process('MyReco')

process.load("FWCore.MessageService.MessageLogger_cfi")

# load the noise info producer
process.load('RecoMET.METProducers.hcalnoiseinfoproducer_cfi')

process.hcalnoise.dropRefVectors = cms.bool(False)
process.hcalnoise.requirePedestals = cms.bool(False)

# run over files
readFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles)
readFiles.extend( [
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_1.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_2.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_7.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_8.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_11.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_12.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_15.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_16.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_17.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_20.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_21.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/MetRecoCalo/R68288calo_22.root'
    ] );

##readFiles.extend( [
##    '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/689151E5-0487-DD11-B613-000423D98800.root'
##    ] );

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(10)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# timing
#process.Timing = cms.Service('Timing')

# for pedestals
#if process.hcalnoise.requirePedestals == True:
#    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#    process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#    process.GlobalTag.globaltag = "CRUZET4_V2P::All"
#    process.prefer("GlobalTag")

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('file:noise_skim.root'),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep recoHcalNoise*_*_*_*',
                                                                      'keep HcalNoiseSummary*_*_*_*',
#                                                                      'keep HBHEDataFramesSorted_*_*_*',
                                                                      'keep HBHERecHitsSorted_*_*_*',
                                                                      'keep recoCaloJets_iterativeCone5CaloJets_*_*',
                                                                      'keep CaloTowersSorted_towerMaker_*_*',
                                                                      'keep recoMET*_*_*_*',
                                                                      'keep recoTracks_generalTracks_*_*')
                               )

  
process.p = cms.Path(process.hcalnoise)

process.e = cms.EndPath(process.out)
