import FWCore.ParameterSet.Config as cms

# Provide dummy calonoise sequences
import FastSimulation.HighLevelTrigger.DummyModule_cfi
HLTHBHENoiseSequence = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
HLTHBHENoiseCleanerSequence = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()



