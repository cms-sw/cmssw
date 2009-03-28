import FWCore.ParameterSet.Config as cms

from RecoTauTag.TauTagTools.TancConditions_cff import *
TauTagMVAComputerRecord.connect = cms.string('frontier://FrontierDev/CMS_COND_BTAU')
