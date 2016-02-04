# The following comments couldn't be translated into the new config version:

#sumETOutputModuleAODSIM &

import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.METSkims.metHigh_OutputModule_cfi import *
from JetMETAnalysis.METSkims.metLow_OutputModule_cfi import *
#include "JetMETAnalysis/METSkims/data/sumET_OutputModule.cfi"
#from JetMETAnalysis.JetSkims.onejet_OutputModule_cfi import *
#from JetMETAnalysis.JetSkims.photonjets_OutputModule_cfi import *
#include "JetMETAnalysis/JetSkims/data/dijetbalance_OutputModule.cfi"
JetMETAnalysisOutput = cms.Sequence(metHighOutputModuleFEVTSIM+metLowOutputModuleAODSIM)#+onejetOutputModuleAODSIM+photonjetsOutputModuleAODSIM)

