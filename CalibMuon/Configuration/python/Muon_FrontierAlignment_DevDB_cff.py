import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_DevDB_cff import *
from CalibMuon.Configuration.Muon_FrontierAlignment_cfi import *
muonAlignment.connect = 'frontier://FrontierPrep/CMS_COND_ALIGNMENT'

