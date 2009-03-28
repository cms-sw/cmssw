import FWCore.ParameterSet.Config as cms

from RecoTauTag.TauTagTools.TancConditions_cff import *
TauTagMVAComputerRecord.connect = cms.string('frontier://FrontierProd/CMS_COND_BTAU')
