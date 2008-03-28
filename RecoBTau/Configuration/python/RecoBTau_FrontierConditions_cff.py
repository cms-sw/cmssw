import FWCore.ParameterSet.Config as cms

from RecoBTau.JetTagComputer.MVAJetTagsFrontierConditions_cfi import *
BTauMVAJetTagComputerRecord.connect = 'frontier://FrontierProd/CMS_COND_20X_BTAU'

