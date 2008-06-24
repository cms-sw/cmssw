import FWCore.ParameterSet.Config as cms

# =====  REPLACE statements for AOD RE-RECO on FAMOS ========
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
# sequence to run for PAT on top of default FastSim 2_0_0 AODSIM event content
patExtraOn200FastSim = cms.Sequence(muIsoDeposits_ParamGlobalMuons * met)

