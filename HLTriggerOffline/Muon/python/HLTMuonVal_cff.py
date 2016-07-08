import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidationHLT_cff import *
# add new muon validation
#from Validation.RecoMuon.NewMuonValidationHLT_cff import *
#
from HLTriggerOffline.Muon.hltMuonValidator_cfi import *

#from DQM.HLTEvF.HLTMonMuonBits_cfi import *
#relvalMuonBits = hltMonMuBits.clone(
#    directory = cms.untracked.string('HLT/Muon'),
#    HLTPaths = cms.vstring('HLT_[^_]*Mu[^l_]*$')
#    )

HLTMuonVal = cms.Sequence(
    recoMuonValidationHLT_seq + 
# to be customized for OLD or NEW muon validation
#    NEWrecoMuonValidationHLT_seq +
    hltMuonValidator
    #+ relvalMuonBits
    )
