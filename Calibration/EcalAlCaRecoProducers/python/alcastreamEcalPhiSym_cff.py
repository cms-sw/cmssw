# The following comments couldn't be translated into the new config version:

#------------------------------------------------
#AlCaReco filtering for phi symmetry calibration:
#------------------------------------------------
#
# Passes events that are coming from the online phi-symmetry stream 
# 
# Id: $Id: alcastreamEcalPhiSym_cff.py,v 1.7 2009/03/26 09:01:53 argiro Exp $
#

import FWCore.ParameterSet.Config as cms

ecalphiSymHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCa_EcalPhiSym*'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False),
TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


