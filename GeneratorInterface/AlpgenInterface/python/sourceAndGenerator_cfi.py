import FWCore.ParameterSet.Config as cms

source = cms.Source("AlpgenSource",
                    fileNames = cms.untracked.vstring('file:w2j')
                    )

from GeneratorInterface.AlpgenInterface.generator_cfi import generator
