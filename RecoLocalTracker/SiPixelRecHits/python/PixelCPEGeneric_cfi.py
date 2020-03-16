import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._generic_default_cfi import _generic_default
PixelCPEGenericESProducer = _generic_default.clone()

# This customizes the Run3 Pixel CPE generic reconstruction in order to activate the IrradiationBiasCorrection
# because of the expected resolution loss due to radiation damage
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(PixelCPEGenericESProducer, IrradiationBiasCorrection = True)

# customize the Pixel CPE generic producer in order not to use any
# template information
from Configuration.ProcessModifiers.phase2_PixelCPEGeneric_cff import phase2_PixelCPEGeneric
phase2_PixelCPEGeneric.toModify(PixelCPEGenericESProducer,
  UseErrorsFromTemplates = False,    # no GenErrors
  LoadTemplatesFromDB = False,       # do not load templates
  TruncatePixelCharge = False,
  IrradiationBiasCorrection = False, # set IBC off (needs GenErrors)
  DoCosmics = False,
  Upgrade = True                     # use hard-coded CPE errors (for Upgrade)
)
