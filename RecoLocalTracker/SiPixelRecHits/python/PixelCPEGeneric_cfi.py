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
  Upgrade = True                     # use 'upgrade' version of hardcoded CPE errors
)

# customize the Pixel CPE generic producer for phase2 3D pixels
# Will remove any usage of template / genError payloads from the reconstruction
from Configuration.Eras.Modifier_phase2_3DPixels_cff import phase2_3DPixels
from Configuration.ProcessModifiers.PixelCPEGeneric_cff import PixelCPEGeneric
(phase2_tracker & (phase2_3DPixels & PixelCPEGeneric)).toModify(PixelCPEGenericESProducer,
                                                                UseErrorsFromTemplates = False,    # no GenErrors
                                                                LoadTemplatesFromDB = False,       # do not load templates
                                                                )

# customize the Pixel CPE generic producer for phase2 square pixels
# Do use Template errors for square pixels even in the first tracking step
# This is needed because hardcoded errors in https://github.com/cms-sw/cmssw/blob/master/RecoLocalTracker/SiPixelRecHits/src/PixelCPEGeneric.cc#L113
# have been optimized for rectangular 25x100 pixels, and in the current generic reco setup we use hardcoded errors for the first tracking pass
from Configuration.Eras.Modifier_phase2_squarePixels_cff import phase2_squarePixels
(phase2_tracker & (phase2_squarePixels | phase2_3DPixels)).toModify(PixelCPEGenericESProducer,
                                                                    NoTemplateErrorsWhenNoTrkAngles = False # use genErrors in the seeding step (when no track angles are available)
                                                                    )
