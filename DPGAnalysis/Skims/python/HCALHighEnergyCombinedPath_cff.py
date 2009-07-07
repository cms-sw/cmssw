import FWCore.ParameterSet.Config as cms
from DPGAnalysis.Skims.HCALHighEnergyHLTPath_cfi import HCALHighEnergyHLTPath
from DPGAnalysis.Skims.HCALHighEnergyFilter_cfi  import HCALHighEnergyFilter

HCALHighEnergyPath = cms.Path( HCALHighEnergyHLTPath + HCALHighEnergyFilter )
