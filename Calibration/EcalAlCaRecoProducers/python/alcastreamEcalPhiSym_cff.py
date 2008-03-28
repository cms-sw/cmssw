# The following comments couldn't be translated into the new config version:

#------------------------------------------------
#AlCaReco filtering for phi symmetry calibration:
#------------------------------------------------
#
# Passes events that are coming from the online phi-symmetry stream 
# 
# Id: $Id: alcastreamEcalPhiSym.cff,v 1.4 2008/03/03 17:05:57 argiro Exp $
#

import FWCore.ParameterSet.Config as cms

ecalphiSymHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('CandHLTEcalPhiSym'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


