import FWCore.ParameterSet.Config as cms

#
# Egamma skim, control sample
#
# reuse the EWK loose Z chain, but with tighter mass cut
from ElectroWeakAnalysis.ZReco.zToEESequences_cff import *
from EgammaAnalysis.CSA07Skims.EgammaLooseZSequence_cff import *
electronRecoForZToEEPath = cms.Path(electronRecoForZToEE)
egammaLooseZTrack = cms.Path(EgammaZOneTrackReco)
egammaLooseZCluster = cms.Path(EgammaZOneSuperClusterReco)

