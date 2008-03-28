import FWCore.ParameterSet.Config as cms

#
# Egamma skim, control sample
#
# reuse the EWK loose Z chain, but with tighter mass cut
# 
from EgammaAnalysis.CSA07Skims.EgammaLowEtTrigger_cfi import *
#
# Redo EWK sequence "electron + track with tighter cut"
#
from EgammaAnalysis.CSA07Skims.EgammaZOneTrackEWKConf_cff import *
from EgammaAnalysis.CSA07Skims.EgammaZOneSuperClusterEWKConf_cff import *
EgammaZOneTrackReco = cms.Sequence(EgammaLowEtTrigger+EgammaZOneTrack+EgammaZOneTrackGenParticlesMatch+EgammaZOneTrackFilter)
EgammaZOneSuperClusterReco = cms.Sequence(EgammaLowEtTrigger+EgammaZOneSuperCluster+EgammaZOneSuperClusterGenParticlesMatch+EgammaZOneSuperClusterFilter)

