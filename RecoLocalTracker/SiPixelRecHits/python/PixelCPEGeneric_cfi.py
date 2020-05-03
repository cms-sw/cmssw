import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._generic_default_cfi import _generic_default
PixelCPEGenericESProducer = _generic_default.clone()

# This customizes the Run3 Pixel CPE generic reconstruction in order to activate the IrradiationBiasCorrection
# because of the expected resolution loss due to radiation damage
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(PixelCPEGenericESProducer, IrradiationBiasCorrection = True)

# customize the Pixel CPE generic producer for phase2
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(PixelCPEGenericESProducer,
  UseErrorsFromTemplates = True,    
  LoadTemplatesFromDB = True,       
  NoTemplateErrorsWhenNoTrkAngles = True,
  TruncatePixelCharge = False,
  IrradiationBiasCorrection = False, # set IBC off
  DoCosmics = False,
  Upgrade = True                     # use hard-coded CPE errors (for Upgrade)
)
