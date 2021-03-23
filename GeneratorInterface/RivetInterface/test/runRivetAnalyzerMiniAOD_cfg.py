import FWCore.ParameterSet.Config as cms

process = cms.Process('Rivet')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

process.load("GeneratorInterface.RivetInterface.mergedGenParticles_cfi")
process.load("GeneratorInterface.RivetInterface.genParticles2HepMC_cff")
process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/relval/CMSSW_10_6_0/RelValTTbar_13/MINIAODSIM/PUpmx25ns_106X_upgrade2018_realistic_v4-v1/10000/DB4F12C2-3B53-4247-A1C3-BEB74F177362.root',
        #'/store/mc/RunIIAutumn18MiniAOD/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v1/00000/0126270E-DB20-FB42-A2B6-CE59183BEB12.root',
        '/store/mc/RunIIAutumn18MiniAOD/TT_noSC_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v1/60000/26D8AE17-B828-C948-A961-A785C6CB71CE.root',
        #'/store/mc/RunIIAutumn18MiniAOD/W2JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v1/120000/1A3A8B55-3655-114D-B933-DA41260548F4.root',
    ),
)

process.options = cms.untracked.PSet()

process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")
process.rivetAnalyzer.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")

process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_2013_I1224539_DIJET', 'MC_TTBAR', 'CMS_2018_I1662081')
process.rivetAnalyzer.OutputFile = cms.string('mcfile.yoda')
process.rivetAnalyzer.useLHEweights = True

process.p = cms.Path(process.mergedGenParticles*process.genParticles2HepMC*process.rivetAnalyzer)
