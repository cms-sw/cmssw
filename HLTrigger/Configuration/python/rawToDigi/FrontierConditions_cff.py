import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.rawToDigi.frontierCablingSetup_cff import *
DTCabling.connect = 'frontier://FrontierProd/CMS_COND_20X_DT'
RPCCabling.connect = 'frontier://FrontierProd/CMS_COND_20X_RPC'
cscPackingCabling.connect = 'frontier://FrontierProd/CMS_COND_20X_CSC'
cscUnpackingCabling.connect = 'frontier://FrontierProd/CMS_COND_20X_CSC'

