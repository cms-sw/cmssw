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

tauNeuralClassifierOnePercent = copy.deepcopy(pfRecoTauDiscriminationByMVAHighEfficiency)
tauNeuralClassifierOnePercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(tauNeuralClassifierOnePercent, CutSet_TaNC_OnePercent)

tauNeuralClassifierOnePercent = copy.deepcopy(pfRecoTauDiscriminationByMVAHighEfficiency)
tauNeuralClassifierOnePercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(tauNeuralClassifierOnePercent, CutSet_TaNC_OnePercent)

tauNeuralClassifierHalfPercent = copy.deepcopy(pfRecoTauDiscriminationByMVAHighEfficiency)
tauNeuralClassifierHalfPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(tauNeuralClassifierHalfPercent, CutSet_TaNC_HalfPercent)

tauNeuralClassifierQuarterPercent = copy.deepcopy(pfRecoTauDiscriminationByMVAHighEfficiency)
tauNeuralClassifierQuarterPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(tauNeuralClassifierQuarterPercent, CutSet_TaNC_QuarterPercent)

tauNeuralClassifierTenthPercent = copy.deepcopy(pfRecoTauDiscriminationByMVAHighEfficiency)
tauNeuralClassifierTenthPercent.MakeBinaryDecision = cms.bool(True)
UpdateCuts(tauNeuralClassifierTenthPercent, CutSet_TaNC_TenthPercent)

RunTanc = cms.Sequence(
      tauNeuralClassifierOnePercent+
      tauNeuralClassifierHalfPercent+
      tauNeuralClassifierQuarterPercent+
      tauNeuralClassifierTenthPercent
      )


