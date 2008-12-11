# The following comments couldn't be translated into the new config version:

#------------------------------------------------
#AlCaReco filtering for phi symmetry calibration:
#------------------------------------------------
#
# Passes events that are coming from the online phi-symmetry stream 
# 
# Id: $Id: alcastreamEcalPhiSym_cff.py,v 1.4 2008/10/30 16:44:38 argiro Exp $
#

import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi

ecalphiSymHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
  HLTPaths = ['AlCa_EcalPhiSym'],
  throw = False,
  andOr = True
  )




