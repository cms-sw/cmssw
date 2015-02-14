# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi

# Adjusted based on
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideAlCaRecoTriggerBits

gammajetHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_L1SingleEG*','HLT_Photon*'],
#   eventSetupPathsKey='HcalCalGammaJet',
    throw = False
)
