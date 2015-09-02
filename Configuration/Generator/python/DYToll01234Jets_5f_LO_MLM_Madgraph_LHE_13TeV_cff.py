import FWCore.ParameterSet.Config as cms 
externalLHEProducer = cms.EDProducer('ExternalLHEProducer', 
                                     scriptName = cms.FileInPath("GeneratorInterface/LHEInterface/data/run_generic_tarball.sh"), 
                                     outputFile = cms.string("cmsgrid_final.lhe"), 
                                     numberOfParameters = cms.uint32(2), 
                                     args = cms.vstring('slc6_amd64_gcc472/13TeV/madgraph/V5_2.2.1/dyellell01234j_5f_LO_MLM/v1/',
                                                        'dyellell01234j_5f_LO_MLM_tarball.tar.gz'),
                                     nEvents = cms.untracked.uint32(10)
                                     ) 
