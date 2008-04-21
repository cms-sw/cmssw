import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.rawToDigi.frontierCablingSetup_cff import *
DTCabling.connect = 'frontier://cms_conditions_data/CMS_COND_20X_DT'
RPCCabling.connect = 'frontier://cms_conditions_data/CMS_COND_20X_RPC'
cscPackingCabling.connect = 'frontier://cms_conditions_data/CMS_COND_20X_CSC'
cscUnpackingCabling.connect = 'frontier://cms_conditions_data/CMS_COND_20X_CSC'

