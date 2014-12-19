import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidationHLT_cff import *
from HLTriggerOffline.Muon.hltMuonValidator_cfi import *
from Validation.RecoMuon.muonValidationHLTFastSim_cff import *

#from DQM.HLTEvF.HLTMonMuonBits_cfi import *
#relvalMuonBits = hltMonMuBits.clone(
#    directory = cms.untracked.string('HLT/Muon'),
#    HLTPaths = cms.vstring('HLT_[^_]*Mu[^l_]*$')
#    )

HLTMuonVal = cms.Sequence(
    recoMuonValidationHLT_seq + 
    hltMuonValidator
    #+ relvalMuonBits
    )

HLTMuonAss_FastSim = cms.Sequence(
    recoMuonAssociationHLTFastSim_seq
    )

HLTMuonVal_FastSim = cms.Sequence(
    recoMuonValidationHLTFastSim_seq +
    hltMuonValidator
    )

