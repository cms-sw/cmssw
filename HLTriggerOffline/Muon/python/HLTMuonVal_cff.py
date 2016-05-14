import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidationHLT_cff import *
# ADD new validation
from Validation.RecoMuon.NewMuonValidationHLT_cff import *
#
from HLTriggerOffline.Muon.hltMuonValidator_cfi import *

#from DQM.HLTEvF.HLTMonMuonBits_cfi import *
#relvalMuonBits = hltMonMuBits.clone(
#    directory = cms.untracked.string('HLT/Muon'),
#    HLTPaths = cms.vstring('HLT_[^_]*Mu[^l_]*$')
#    )

# to be customized for OLD or NEW validation
HLTMuonVal = cms.Sequence(
#    recoMuonValidationHLT_seq + 
    NEWrecoMuonValidationHLT_seq +
#
    hltMuonValidator
    #+ relvalMuonBits
    )
