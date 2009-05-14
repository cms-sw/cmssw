import FWCore.ParameterSet.Config as cms
from DPGAnalysis.Skims.HCALHighEnergyHLTPath_cfi   import HCALHighEnergyHLTPath
from DPGAnalysis.Skims.HCALHighEnergyFilter_cfi    import HCALHighEnergyFilter
from DPGAnalysis.Skims.HCALHighEnergyHPDFilter_cfi import HCALHighEnergyHPDFilter

HCALHighEnergyPath = cms.Path( HCALHighEnergyHLTPath * HCALHighEnergyHPDFilter * HCALHighEnergyFilter )
