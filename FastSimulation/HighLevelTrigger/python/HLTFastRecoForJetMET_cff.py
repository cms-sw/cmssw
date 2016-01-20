import FWCore.ParameterSet.Config as cms

# Provide dummy calonoise sequences
import FastSimulation.HighLevelTrigger.DummyModule_cfi
# LV: probably no need to replace this with a dummy
HLTHBHENoiseCleanerSequence = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()



