import FWCore.ParameterSet.Config as cms
import copy

#Define the mapping of Decay mode IDs onto the names of trained MVA files
#Note that while one category can apply to multiple decay modes, a decay mode can not have multiple categories

# Get MVA configuration defintions (edit MVAs here)
from RecoTauTag.TauTagTools.TauMVAConfigurations_cfi import *
from RecoTauTag.TauTagTools.TauMVADiscriminator_cfi import *
from RecoTauTag.TauTagTools.BenchmarkPointCuts_cfi import *

def UpdateCuts(TheProducer, TheCuts):
   for aComputer in TheProducer.computers:
      TheCut = TheCuts[aComputer.computerName.value()]
      # By default TMVA produces an output from [-1, 1].  The benchmark points are computed using this range.
      #  if the user has specified to remap the output to [0, 1] (usual case), remap the benchmark point used 
      #  to produce the binary decision
      if TheProducer.RemapOutput:
         TheCut += 1.0
         TheCut /= 2.0
      aComputer.cut = cms.double(TheCut)

TauDecayModeCutMutliplexerPrototype = cms.EDProducer("PFTauDecayModeCutMultiplexer",
      PFTauDecayModeSrc            = cms.InputTag("shrinkingConePFTauDecayModeIndexProducer"),
      PFTauDiscriminantToMultiplex = cms.InputTag('shrinkingConePFTauDiscriminationByTaNC'),
      preDiscriminants             = cms.VInputTag("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),
      computers                    = TaNC,
      RemapOutput                  = cms.bool(True)
)

shrinkingConePFTauDiscriminationByTaNCfrOnePercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
shrinkingConePFTauDiscriminationByTaNCfrOnePercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrOnePercent, CutSet_TaNC_OnePercent)

shrinkingConePFTauDiscriminationByTaNCfrOnePercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
shrinkingConePFTauDiscriminationByTaNCfrOnePercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrOnePercent, CutSet_TaNC_OnePercent)

shrinkingConePFTauDiscriminationByTaNCfrHalfPercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
shrinkingConePFTauDiscriminationByTaNCfrHalfPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrHalfPercent, CutSet_TaNC_HalfPercent)

shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent, CutSet_TaNC_QuarterPercent)

shrinkingConePFTauDiscriminationByTaNCfrTenthPercent = copy.deepcopy(TauDecayModeCutMutliplexerPrototype)
shrinkingConePFTauDiscriminationByTaNCfrTenthPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrTenthPercent, CutSet_TaNC_TenthPercent)

RunTanc = cms.Sequence(
      shrinkingConePFTauDiscriminationByTaNCfrOnePercent+
      shrinkingConePFTauDiscriminationByTaNCfrHalfPercent+
      shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent+
      shrinkingConePFTauDiscriminationByTaNCfrTenthPercent
      )


