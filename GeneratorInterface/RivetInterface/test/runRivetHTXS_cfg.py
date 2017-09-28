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

# just run a lot of MINIAOD
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/2C7F3153-393B-E611-9323-0CC47AA98A3A.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/32B22B7F-393B-E611-A66F-002618FDA287.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/3A6222E9-393B-E611-BE44-A0000420FE80.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/5C03BAED-393B-E611-95C9-0242AC110002.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/6C09D968-393B-E611-968F-D067E5F91E51.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/B04A1D54-393B-E611-A8EE-00266CF3E3C4.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/B63E6954-393B-E611-AEF5-3417EBE706C6.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/BA38B07D-393B-E611-9CEB-0CC47A4D7600.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/BE84118B-393B-E611-B14D-0025905B85A0.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/E661CC7A-393B-E611-BD13-001E67A40523.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/EC99CC47-513B-E611-A22E-0242AC130002.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/F2A54B6A-393B-E611-9726-0242AC130004.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/F41CA385-393B-E611-9DF7-0CC47A4D767E.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/10000/FC8DA0B9-393B-E611-B9B8-003048CB8768.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/0A95C225-A53B-E611-B926-0025905B85BC.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/143DCA57-963B-E611-9B03-0CC47A78A3B4.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/28692A21-A53B-E611-A032-0CC47A4D7600.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/424E7EC7-B53B-E611-AE1C-0CC47A4D7628.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/482AD8ED-EE3B-E611-9562-0025905A608E.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/4C487348-F03B-E611-ADFE-0242AC110002.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/50EC21CB-DA3B-E611-B715-00266CF3E4B8.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/648220ED-8D3B-E611-B790-D4AE52E5DE74.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/6C0981CF-933B-E611-A684-003048FFD7CE.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/6E1D5FDB-AC3B-E611-9359-0CC47A4C8ECE.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/96DE9BA1-D93B-E611-B451-002590D425C0.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/AC618789-9F3B-E611-B7AA-0CC47A4C8F18.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/AEA6668A-AA3B-E611-A694-0CC47A78A42E.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/AEF809C5-933B-E611-98DA-0CC47A78A408.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/E03174B8-D93B-E611-95A3-00266CF3E4B8.root',
'/store/mc/RunIISpring16MiniAODv2/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/80000/EE6619A0-963B-E611-B4F9-0CC47A78A408.root',

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
process.myGenerator = cms.EDProducer("GenParticles2HepMCConverterHTXS",
    genParticles = cms.InputTag("mergedGenParticles"),
    genEventInfo = cms.InputTag("generator"),
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
