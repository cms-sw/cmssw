import FWCore.ParameterSet.Config as cms

process = cms.Process('MyReco')

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('RecoMET.METProducers.hcalnoiseinfoproducer_cfi')

# run over files
readFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles)
readFiles.extend( [
    'file:/uscms_data/d2/lpcjm/CRAFT/CALO/R68288calo_1.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/CALO/R68288calo_2.root',
    'file:/uscms_data/d2/lpcjm/CRAFT/CALO/R68288calo_4.root'
    ] );

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(10)

# timing
#process.Timing = cms.Service('Timing')

# for pedestals
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V2P::All"
process.prefer("GlobalTag")

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('test.root'),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep recoHcalNoise*_*_*_*',
                                                                      'keep recoMET*_*_*_*')
                               )

  
process.p = cms.Path(process.hcalnoiseinfoproducer)

process.e = cms.EndPath(process.out)
