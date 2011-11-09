import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer('ExternalLHEProducer',
    scriptName = cms.FileInPath("GeneratorInterface/LHEInterface/data/run_madgraph_gridpack.sh"),
    outputFile = cms.string("events.lhe"),
    args = cms.vstring('slc5_ia32_gcc434/madgraph/V5_1.1/7TeV_Summer11/Gridpacks','W1jets','100','false','true','wjets','5','20','false','0','99')
)

