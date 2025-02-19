import FWCore.ParameterSet.Config as cms
import copy

#Define the mapping of Decay mode IDs onto the names of trained MVA files
#Note that while one category can apply to multiple decay modes, a decay mode can not have multiple categories

# Get MVA configuration defintions (edit MVAs here)
from RecoTauTag.TauTagTools.TauMVAConfigurations_cfi import *
from RecoTauTag.TauTagTools.TauMVADiscriminator_cfi import *
from RecoTauTag.TauTagTools.BenchmarkPointCuts_cfi import *

# Temporary
dmCodeTrans = {
    (1,0) : 'OneProngNoPiZero',
    (1,1) : 'OneProngOnePiZero',
    (1,2) : 'OneProngTwoPiZero',
    (3,0) : 'ThreeProngNoPiZero',
    (3,1) : 'ThreeProngOnePiZero',
}

def UpdateCuts(producer, cut_set):
    oldDecayModes = producer.decayModes
    newDecayModes = cms.VPSet()
    for dm in oldDecayModes:
        cut = cut_set[dmCodeTrans[(dm.nCharged.value(), dm.nPiZeros.value())]]
        cut += 1.0
        cut /= 2.0
        newDecayMode = copy.deepcopy(dm)
        newDecayMode.cut = cms.double(cut)
        newDecayModes.append(newDecayMode)
    producer.decayModes = newDecayModes

TauDecayModeCutMutliplexerPrototype = cms.EDProducer(
    "RecoTauDecayModeCutMultiplexer",
    PFTauProducer = cms.InputTag("shrinkingConePFTauProducer"),
    toMultiplex = cms.InputTag('shrinkingConePFTauDiscriminationByTaNC'),
    Prediscriminants = shrinkingConeLeadTrackFinding,
    #computers = TaNC,
    decayModes = cms.VPSet(
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(0),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(1),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(2),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(0),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(1),
        ),
    )
)

shrinkingConePFTauDiscriminationByTaNCfrOnePercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrOnePercent, CutSet_TaNC_OnePercent)

shrinkingConePFTauDiscriminationByTaNCfrOnePercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrOnePercent, CutSet_TaNC_OnePercent)

shrinkingConePFTauDiscriminationByTaNCfrHalfPercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrHalfPercent, CutSet_TaNC_HalfPercent)

shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent, CutSet_TaNC_QuarterPercent)

shrinkingConePFTauDiscriminationByTaNCfrTenthPercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrTenthPercent, CutSet_TaNC_TenthPercent)

RunTanc = cms.Sequence(
      shrinkingConePFTauDiscriminationByTaNCfrOnePercent+
      shrinkingConePFTauDiscriminationByTaNCfrHalfPercent+
      shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent+
      shrinkingConePFTauDiscriminationByTaNCfrTenthPercent
      )


