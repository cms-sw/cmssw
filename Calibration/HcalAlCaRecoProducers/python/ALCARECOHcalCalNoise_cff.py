import FWCore.ParameterSet.Config as cms


import HLTrigger.HLTfilters.hltHighLevel_cfi
noiseHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_MET25'],
#    eventSetupPathsKey='HcalCalNoise',
    throw = False #dont throw except on unknown path name
)

prescaler = cms.EDFilter("PrescalerFHN",
TriggerResultsTag = cms.InputTag("TriggerResults", "", "HLT"),

# Will select OR of all specified HLTs
# And increment if HLT is seen, regardless if
# others cause selection

Prescales = cms.VPSet(
    cms.PSet(
    HLTName = cms.string("HLT_MET25"),
    PrescaleFactor = cms.uint32(1)
    )
#    cms.PSet(
#    name = cms.string("HLTPath2"),
#    factor = cms.uint32(100)
#    )
))

from Configuration.StandardSequences.RawToDigi_cff import *

from Configuration.StandardSequences.Reconstruction_cff import *

from Calibration.HcalAlCaRecoProducers.alcahcalnoise_cfi import * 

kt4CaloJets.correctInputToSignalVertex = False
kt6CaloJets.correctInputToSignalVertex = False
iterativeCone5CaloJets.correctInputToSignalVertex = False
sisCone5CaloJets.correctInputToSignalVertex = False
sisCone7CaloJets.correctInputToSignalVertex = False

doNoiseDigi=cms.Sequence(ecalDigis+ecalPreshowerDigis+hcalDigis)

doNoiseLocalReco=cms.Sequence(pixeltrackerlocalreco + calolocalreco)

doNoiseGlobalReco=cms.Sequence(caloTowersRec*recoJets + metreco)

seqALCARECOHcalCalNoise = cms.Sequence(noiseHLT*prescaler*doNoiseDigi*doNoiseLocalReco*doNoiseGlobalReco*HcalNoiseProd)


