import FWCore.ParameterSet.Config as cms

# FIXME: GlobalPosition_Frontier_IntDB.cff does not (yet) exist
from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_IntDB_cff import *
from CalibMuon.Configuration.Muon_FrontierAlignment_cfi import *
muonAlignment.connect = 'frontier://cms_conditions_data/CMS_COND_20X_ALIGNMENT'

