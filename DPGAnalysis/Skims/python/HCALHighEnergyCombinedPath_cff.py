import FWCore.ParameterSet.Config as cms

# it does not work in 3_1_1 trigger table
#from DPGAnalysis.Skims.HCALHighEnergyHLTPath_cfi   import HCALHighEnergyHLTPath
from DPGAnalysis.Skims.HCALHighEnergyHPDFilter_cfi import HCALHighEnergyHPDFilter
from DPGAnalysis.Skims.HCALHighEnergyFilter_cfi    import HCALHighEnergyFilter

#HCALHighEnergyPath = cms.Path( HCALHighEnergyHLTPath * HCALHighEnergyHPDFilter * HCALHighEnergyFilter )
HCALHighEnergyPath = cms.Path( HCALHighEnergyHPDFilter * HCALHighEnergyFilter )
