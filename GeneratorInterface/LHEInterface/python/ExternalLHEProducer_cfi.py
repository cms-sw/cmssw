import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer('ExternalLHEProducer',
    scriptName = cms.FileInPath("GeneratorInterface/LHEInterface/data/run_madgraph_gridpack.sh"),
    outputFile = cms.string("W1Jet_7TeV_madgraph_final.lhe"),
    numberOfParameters = cms.uint32(10),
    args = cms.vstring('slc5_ia32_gcc434/madgraph/V5_1.3.27/test','W1Jet_7TeV_madgraph','false','true','wjets','5','20','false','0','99'),
    nEvents = cms.uint32(100)                                    
)

