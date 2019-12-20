import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._templates_default_cfi import _templates_default
templates = _templates_default.clone()

from Configuration.ProcessModifiers.run4_PixelCPEGeneric_cff import run4_PixelCPEGeneric
run4_PixelCPEGeneric.toModify(templates,
  LoadTemplatesFromDB = False,
  DoLorentz = False,
)
