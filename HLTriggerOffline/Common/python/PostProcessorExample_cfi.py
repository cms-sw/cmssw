#the sequence defined herein do not get included
# in the actual sequences run centrally
#these module instances serve as examples of postprocessing
# new histograms are generated as the ratio of the specified input
# defined thru 'efficiency = cms.vstring(arg)' where arg has the form
# "ratio 'histo title; x-label; y-label; numerator denominator'"
#the base code is in: DQMServices/ClientConfig/plugins/DQMGenericClient.cc
#note: output and verbose must be disabled for integration,

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

myMuonPostVal = DQMEDHarvester("DQMGenericClient",
    verbose        = cms.untracked.uint32(0), #set this to zero!
    outputFileName = cms.untracked.string(''),# set this to empty!
    #outputFileName= cms.untracked.string('MuonPostProcessor.root'),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),                                    
    subDirs        = cms.untracked.vstring('HLT/Muon/Distributions/*'),
    efficiency     = cms.vstring(
        "EFF 'my title; my x-label; my y-label' genPassEta_L1Filtered genPassEta_All"
    )
)


myEgammaPostVal = DQMEDHarvester("DQMGenericClient",
    #outputFileName= cms.untracked.string('EgammaPostProcessor.root'),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),                                    
    subDirs        = cms.untracked.vstring('HLT/HLTEgammaValidation/*'),
    efficiency     = cms.vstring(
        "EFF 'my title; my x-label; my y-label' hltL1sDoubleEgammaeta hltL1sDoubleEgammaeta"
    )
)

myTauPostVal = DQMEDHarvester("DQMGenericClient",
    #outputFileName= cms.untracked.string('TauPostProcessor.root'),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),                                    
    subDirs        = cms.untracked.vstring('HLT/HLTTAU/*'),
    efficiency     = cms.vstring(
        "EFF 'my title; my x-label; my y-label' L1Tau1Eta GenTauElecEta"
    )
)

ExamplePostVal = cms.Sequence(
     myMuonPostVal
    +myEgammaPostVal
    +myTauPostVal
)
