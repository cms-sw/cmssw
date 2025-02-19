import FWCore.ParameterSet.Config as cms

from RecoBTau.JetTagComputer.MVAJetTagsFrontierConditions_cfi import *
BTauMVAJetTagComputerRecord.connect = 'frontier://FrontierDev/CMS_COND_BTAU'

