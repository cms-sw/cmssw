import FWCore.ParameterSet.Config as cms

from Configuration.JetMET.CaloConditions_cff import *
from SimGeneral.MixingModule.mixNoPU_cfi import *
from SimCalorimetry.Configuration.SimCalorimetry_cff import *
caloDigi = cms.Sequence(mix*calDigi)

