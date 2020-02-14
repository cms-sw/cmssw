import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._templates2_default_cfi import _templates2_default
templates2 = _templates2_default.clone()

from Configuration.ProcessModifiers.run4_PixelCPEGeneric_cff import run4_PixelCPEGeneric
run4_PixelCPEGeneric.toModify(templates2,
                              LoadTemplatesFromDB = False,
                              DoLorentz = False,
                              )
