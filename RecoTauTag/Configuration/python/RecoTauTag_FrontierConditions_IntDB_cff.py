import FWCore.ParameterSet.Config as cms

from RecoTauTag.TauTagTools.TancConditions_cff import *
TauTagMVAComputerRecord.connect = cms.string('frontier://cms_conditions_data/CMS_COND_20X_BTAU')
