import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4CaloJets_cfi import *
from ..sequences.caloTowersRec_cfi import *
from ..sequences.ecalClusters_cfi import *
from ..sequences.globalreco_tracking_cfi import *
from ..sequences.hcalGlobalRecoSequence_cfi import *
from ..sequences.hgcalLocalRecoSequence_cfi import *
from ..sequences.iterTICLSequence_cfi import *
from ..sequences.muonGlobalReco_cfi import *
from ..sequences.particleFlowCluster_cfi import *
from ..sequences.standalonemuontracking_cfi import *

globalreco = cms.Sequence(caloTowersRec+ecalClusters+hcalGlobalRecoSequence+globalreco_tracking+standalonemuontracking+hgcalLocalRecoSequence+cms.Sequence(cms.Task())+hltAK4CaloJets+muonGlobalReco+particleFlowCluster+hltAK4CaloJets+iterTICLSequence)
