import FWCore.ParameterSet.Config as cms
import copy

#Define the mapping of Decay mode IDs onto the names of trained MVA files
#Note that one category can apply to multiple decay modes, a decay mode can not have multiple categories

# Get MVA configuration defintions (edit MVAs here)
from RecoTauTag.TauTagTools.TauMVAConfigurations_cfi import *
from RecoTauTag.TauTagTools.TauMVADiscriminator_cfi import *
from RecoTauTag.TauTagTools.BenchmarkPointCuts_cfi import *

def UpdateCuts(TheProducer, TheCuts):
   for aComputer in TheProducer.computers:
      aComputer.cut = cms.double(TheCuts[aComputer.computerName.value()])

shrinkingConePFTauDiscriminationByTaNCfrOnePercent = copy.deepcopy(shrinkingConePFTauDiscriminationByTaNC)
shrinkingConePFTauDiscriminationByTaNCfrOnePercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrOnePercent, CutSet_TaNC_OnePercent)

shrinkingConePFTauDiscriminationByTaNCfrOnePercent = copy.deepcopy(shrinkingConePFTauDiscriminationByTaNC)
shrinkingConePFTauDiscriminationByTaNCfrOnePercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrOnePercent, CutSet_TaNC_OnePercent)

shrinkingConePFTauDiscriminationByTaNCfrHalfPercent = copy.deepcopy(shrinkingConePFTauDiscriminationByTaNC)
shrinkingConePFTauDiscriminationByTaNCfrHalfPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrHalfPercent, CutSet_TaNC_HalfPercent)

shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent = copy.deepcopy(shrinkingConePFTauDiscriminationByTaNC)
shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent, CutSet_TaNC_QuarterPercent)

shrinkingConePFTauDiscriminationByTaNCfrTenthPercent = copy.deepcopy(shrinkingConePFTauDiscriminationByTaNC)
shrinkingConePFTauDiscriminationByTaNCfrTenthPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(shrinkingConePFTauDiscriminationByTaNCfrTenthPercent, CutSet_TaNC_TenthPercent)

RunTanc = cms.Sequence(
      shrinkingConePFTauDiscriminationByTaNCfrOnePercent+
      shrinkingConePFTauDiscriminationByTaNCfrHalfPercent+
      shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent+
      shrinkingConePFTauDiscriminationByTaNCfrTenthPercent
      )


