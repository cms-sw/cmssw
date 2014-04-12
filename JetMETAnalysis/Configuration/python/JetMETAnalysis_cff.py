import FWCore.ParameterSet.Config as cms

#
# JetMETAnalysis standard sequences
#
from JetMETAnalysis.METSkims.METSkims_cff import *
#from JetMETAnalysis.JetSkims.JetSkims_cff import *
jetMETAnalysis = cms.Sequence(metSkims)

