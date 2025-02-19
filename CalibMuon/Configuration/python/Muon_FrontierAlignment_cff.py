import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff import *
from CalibMuon.Configuration.Muon_FrontierAlignment_cfi import *
muonAlignment.connect = 'frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'

