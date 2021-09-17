import FWCore.ParameterSet.Config as cms
process = cms.Process("runRivetAnalysis")

process.options   = cms.untracked.PSet(                           
    allowUnscheduled = cms.untracked.bool(False)
) 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",  fileNames = cms.untracked.vstring(
# compare AOD and MINIAOD
#'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/2C7F3153-393B-E611-9323-0CC47AA98A3A.root'
#'/store/mc/RunIISpring16reHLT80/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/AODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/52E144D7-793A-E611-B70F-0025904A8ECC.root',
#'/store/mc/RunIISpring16reHLT80/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/AODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/D650466F-A13A-E611-AA11-0CC47A13CD56.root'

# just run some MINIAOD
'/store/mc/RunIIAutumn18MiniAOD/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUGenV7011_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/80000/6F4F411E-8111-684D-827D-B5962A0CB94F.root',
'/store/mc/RunIIAutumn18MiniAOD/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUGenV7011_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/80000/49CB36B2-E124-2249-A0F8-CE867CF4F8A6.root',
'/store/mc/RunIIAutumn18MiniAOD/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUGenV7011_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/80000/79F49EC1-42B4-3349-A268-59510E899BCC.root',
'/store/mc/RunIIAutumn18MiniAOD/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUGenV7011_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/80000/D65A4D51-2E80-AD41-B50D-E4083BA2A668.root',

),
)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.rivetProducerHTXS = cms.EDProducer('HTXSRivetProducer',
  HepMCCollection = cms.InputTag('myGenerator','unsmeared'),
  LHERunInfo = cms.InputTag('externalLHEProducer'),
  #ProductionMode = cms.string('GGF'),
  ProductionMode = cms.string('AUTO'),
)

#MINIAOD
process.mergedGenParticles = cms.EDProducer("MergedGenParticleProducer",
    inputPruned = cms.InputTag("prunedGenParticles"),
    inputPacked = cms.InputTag("packedGenParticles"),
)
process.myGenerator = cms.EDProducer("GenParticles2HepMCConverter",
    genParticles = cms.InputTag("mergedGenParticles"),
    genEventInfo = cms.InputTag("generator"),
    signalParticlePdgIds = cms.vint32(25), ## for the Higgs analysis
)
process.p = cms.Path(process.mergedGenParticles*process.myGenerator*process.rivetProducerHTXS)

# # AOD
#process.myGenerator = cms.EDProducer("GenParticles2HepMCConverterHTXS",
#    genParticles = cms.InputTag("genParticles"),
#    genEventInfo = cms.InputTag("generator"),
#)
#process.p = cms.Path(process.myGenerator*process.rivetProducerHTXS)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *','keep *_*_*_runRivetAnalysis','keep *_generator_*_*','keep *_externalLHEProducer_*_*'),
    fileName = cms.untracked.string('testHTXSRivet_ggH4l_MINIAOD_100k.root')
)
process.o = cms.EndPath( process.out )
