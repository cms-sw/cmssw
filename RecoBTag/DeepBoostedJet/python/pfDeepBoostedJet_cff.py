import FWCore.ParameterSet.Config as cms

from RecoBTag.DeepBoostedJet.pfDeepBoostedJetTagInfos_cfi import pfDeepBoostedJetTagInfos
from RecoBTag.DeepBoostedJet.pfDeepBoostedJetTags_cfi import _pfDeepBoostedJetTags
from RecoBTag.DeepBoostedJet.pfDeepBoostedJetPreprocessParams_cfi import pfDeepBoostedJetPreprocessParams
from RecoBTag.DeepBoostedJet.pfDeepBoostedDiscriminatorsJetTags_cfi import pfDeepBoostedDiscriminatorsJetTags
from RecoBTag.DeepBoostedJet.pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags_cfi import pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags

# nominal DeepAK8
pfDeepBoostedJetTags = _pfDeepBoostedJetTags.clone(
    preprocessParams = pfDeepBoostedJetPreprocessParams,
    model_path = cms.FileInPath('RecoBTag/Combined/data/DeepBoostedJet/V01/full/resnet-symbol.json'),
    param_path = cms.FileInPath('RecoBTag/Combined/data/DeepBoostedJet/V01/full/resnet-0000.params'),
    debugMode = cms.untracked.bool(False), # debug
)

# mass-decorrelated DeepAK8
pfMassDecorrelatedDeepBoostedJetTags = _pfDeepBoostedJetTags.clone(
    preprocessParams = pfDeepBoostedJetPreprocessParams,
    model_path = cms.FileInPath('RecoBTag/Combined/data/DeepBoostedJet/V01/decorrelated/resnet-symbol.json'),
    param_path = cms.FileInPath('RecoBTag/Combined/data/DeepBoostedJet/V01/decorrelated/resnet-0000.params'),
    debugMode = cms.untracked.bool(False), # debug
)

from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run DeepFlavour from RECO
# jets (RECO/AOD)
pfDeepBoostedJetTask = cms.Task(puppi, primaryVertexAssociation,
                             pfDeepBoostedJetTagInfos, pfDeepBoostedJetTags, pfMassDecorrelatedDeepBoostedJetTags)
