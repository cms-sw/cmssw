import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBtagDeepCSVSequencePF_cfi import *

noFilter_PFDeepCSV_path = cms.Path(HLTBtagDeepCSVSequencePF)
