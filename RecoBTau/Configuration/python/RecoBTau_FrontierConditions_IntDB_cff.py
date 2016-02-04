import FWCore.ParameterSet.Config as cms

from RecoBTau.JetTagComputer.MVAJetTagsFrontierConditions_cfi import *
BTauMVAJetTagComputerRecord.connect = 'frontier://cms_conditions_data/CMS_COND_20X_BTAU'

