import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton

#behind-the-scenes modifier that only turns on the DRN photon regression
_photonDRN = cms.Modifier()

#modifier to enable DRN energy regression for photons
#requires also having enableSonicTriton
photonDRN = cms.ModifierChain(_photonDRN, enableSonicTriton)
