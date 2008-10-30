# The following comments couldn't be translated into the new config version:

#------------------------------------------------
#AlCaReco filtering for phi symmetry calibration:
#------------------------------------------------
#
# Passes events that are coming from the online phi-symmetry stream 
# 
# Id: $Id: alcastreamEcalPhiSym_cff.py,v 1.3 2008/06/14 10:33:43 elmer Exp $
#

import FWCore.ParameterSet.Config as cms

ecalphiSymHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCa_EcalPhiSym'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), #dont throw except on unknown path name 
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


