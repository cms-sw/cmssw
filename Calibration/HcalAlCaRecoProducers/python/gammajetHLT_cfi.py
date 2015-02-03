# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi

gammajetHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_L1SingleEG10','HLT_L1SingleEG12',
                'HLT_L1SingleEG20',
                # from 2012 menu
                'HLT_Photon20_CaloIdVL_IsoL',
                'HLT_Photon30_CaloIdVL_IsoL',
                'HLT_Photon50_CaloIdVL_IsoL',
                'HLT_Photon75_CaloIdVL_IsoL',
                'HLT_Photon90_CaloIdVL_IsoL',
                'HLT_Photon135',
                'HLT_Photon150',
                'HLT_Photon160',
                # from CMSSW 7_4_0_pre2
                'HLT_Photon22','HLT_Photon30', 'HLT_Photon36',
                'HLT_Photon50', 'HLT_Photon75',
                'HLT_Photon90', 'HLT_Photon120', 'HLT_Photon175',
                'HLT_Photon250_NoHE', 'HLT_Photon300_NoHE'
    ],
    throw = False
)


