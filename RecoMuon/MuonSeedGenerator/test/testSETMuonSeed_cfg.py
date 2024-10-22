import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoMuon")

# Messages
#process.load("RecoMuon.Configuration.MessageLogger_cfi")
#process.load("FWCore.MessageService.MessageLogger_cfi")

# Muon Reco
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

process.load("RecoMuon.Configuration.RecoMuon_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1000)
            )
process.source = cms.Source("PoolSource",
                                    fileNames = cms.untracked.vstring(

    #'/store/relval/CMSSW_3_0_0_pre7/RelValSingleMuPt100/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0006/50187155-5EE8-DD11-BF56-000423D9517C.root'
    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_30X_v1/0006/4C2DF180-6AE8-DD11-B789-000423D99AA2.root'
    #"dcache:/pnfs/cms/WAX/resilient/ibloch/CRAB_output/MuID_samples_3_0/3_0_0_pre3/mumin_e_60_300_probev2__1.root"
    ),
                                    skipEvents = cms.untracked.uint32(0)
                            )



process.out = cms.OutputModule("PoolOutputModule",
                                                                fileName = cms.untracked.string('RecoMuons.root')
                                                                )

  ## SET algorithm of STA muon
  #from RecoMuon.SeedGenerator.selectorSET_cff import *
process.STAMuonAnalyzer = cms.EDAnalyzer("STAMuonAnalyzer",
                                         #    DataType = cms.untracked.string('SimData'),
                                             DataType = cms.untracked.string('RealData'),
                                             StandAloneTrackCollectionLabel = cms.untracked.string('standAloneSETMuons'),
                                         #    MuonSeedCollectionLabel = cms.untracked.string('MuonSeedTester'),
                                             MuonSeedCollectionLabel = cms.untracked.string('SETMuonSeed'),

                                             rootFileName = cms.untracked.string('STAMuonAnalyzer.root'),
                                         )


process.p = cms.Path(process.SETMuonSeed*process.standAloneSETMuons*process.STAMuonAnalyzer)
  #process.this_is_the_end = cms.EndPath(process.out)

process.GlobalTag.globaltag = 'IDEAL_30X::All'
  #process.GlobalTag.globaltag = 'STARTUP_30X::All'
  

  
