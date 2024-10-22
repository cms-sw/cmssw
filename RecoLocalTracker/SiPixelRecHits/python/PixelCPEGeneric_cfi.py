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
  isPhase2 = True,                    # use 'Phase2' version of hardcoded CPE errors
  xerr_barrel_ln = [0.00025, 0.00030, 0.00035, 0.00035],
  xerr_barrel_ln_def = 0.00035,
  yerr_barrel_ln = [0.00210, 0.00115, 0.00125],
  yerr_barrel_ln_def = 0.00125,
  xerr_endcap = [0.00072, 0.00025],
  xerr_endcap_def = 0.00060,
  yerr_endcap = [0.00289, 0.00025],
  yerr_endcap_def = 0.00180,
  # if SmallPitch
  # xerr_barrel_l1 = [0.00104, 0.000691, 0.00122],
  # xerr_barrel_l1_def = 0.00321,
  # yerr_barrel_l1 = [0.00199, 0.00136, 0.0015, 0.00153, 0.00152, 0.00171, 0.00154, 0.00157, 0.00154],
  # yerr_barrel_l1_def = 0.00164
  # else
  xerr_barrel_l1 = [0.00025, 0.00030, 0.00035, 0.00035],
  xerr_barrel_l1_def = 0.00035,
  yerr_barrel_l1 = [0.00210, 0.00115, 0.00125],
  yerr_barrel_l1_def = 0.00125
)

# customize the Pixel CPE generic producer for phase2 3D pixels
# Will remove any usage of template / genError payloads from the reconstruction
from Configuration.Eras.Modifier_phase2_3DPixels_cff import phase2_3DPixels
from Configuration.ProcessModifiers.PixelCPEGeneric_cff import PixelCPEGeneric
(phase2_tracker & (phase2_3DPixels & PixelCPEGeneric)).toModify(PixelCPEGenericESProducer,
                                                                UseErrorsFromTemplates = False,    # no GenErrors
                                                                LoadTemplatesFromDB = False        # do not load templates
                                                                )
# customize the Pixel CPE generic producer for phase2 square pixels
# Do use Template errors for square pixels even in the first tracking step
# This is needed because hardcoded errors in https://github.com/cms-sw/cmssw/blob/master/RecoLocalTracker/SiPixelRecHits/src/PixelCPEGeneric.cc#L113
# have been optimized for rectangular 25x100 pixels, and in the current generic reco setup we use hardcoded errors for the first tracking pass
from Configuration.Eras.Modifier_phase2_squarePixels_cff import phase2_squarePixels
(phase2_tracker & (phase2_squarePixels | phase2_3DPixels)).toModify(PixelCPEGenericESProducer,
                                                                    NoTemplateErrorsWhenNoTrkAngles = False # use genErrors in the seeding step (when no track angles are available)
                                                                    )
