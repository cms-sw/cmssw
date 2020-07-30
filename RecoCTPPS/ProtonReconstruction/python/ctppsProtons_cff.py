import FWCore.ParameterSet.Config as cms

# import default alignment settings
from CalibPPS.ESProducers.ctppsAlignment_cff import *

# import default optics settings
from CalibPPS.ESProducers.ctppsOpticalFunctions_cff import *

# import and adjust proton-reconstructions settings
from RecoCTPPS.ProtonReconstruction.ctppsProtons_cfi import *
ctppsProtons.lhcInfoLabel = ctppsLHCInfoLabel

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL

def applyDefaultSettings(ctppsProtons):
  ctppsProtons.association_cuts_45.x_cut_apply = False
  ctppsProtons.association_cuts_45.y_cut_apply = False
  ctppsProtons.association_cuts_45.xi_cut_apply = True
  ctppsProtons.association_cuts_45.xi_cut_value = 0.010
  ctppsProtons.association_cuts_45.th_y_cut_apply = False

  ctppsProtons.association_cuts_56.x_cut_apply = False
  ctppsProtons.association_cuts_56.y_cut_apply = False
  ctppsProtons.association_cuts_56.xi_cut_apply = True
  ctppsProtons.association_cuts_56.xi_cut_value = 0.015
  ctppsProtons.association_cuts_56.th_y_cut_apply = False

  # update for re-miniAOD
  run2_miniAOD_UL.toModify(ctppsProtons,
    pixelDiscardBXShiftedTracks = True,
    association_cuts_45 = dict(ti_tr_min = -1.5, ti_tr_max = 2.0),
    association_cuts_56 = dict(ti_tr_min = -1.5, ti_tr_max = 2.0),
    default_time = -999.
  )

ctpps_2016.toModify(ctppsProtons, applyDefaultSettings) # applied for all Run2 years (2016, 2017 and 2018)

def apply2017Settings(ctppsProtons):
  ctppsProtons.association_cuts_45.xi_cut_value = 0.010
  ctppsProtons.association_cuts_56.xi_cut_value = 0.015

  # update for re-miniAOD
  run2_miniAOD_UL.toModify(ctppsProtons,
    association_cuts_45 = dict(
      x_cut_apply = False,
      y_cut_apply = False,

      xi_cut_apply = True,
      xi_cut_value = 5. * 0.00121,
      xi_cut_mean = +6.0695e-5,

      th_y_cut_apply = False
    ),

    association_cuts_56 = dict(
      x_cut_apply = False,

      y_cut_apply = True,
      y_cut_value = 5. * 0.14777,
      y_cut_mean = -0.022612,

      xi_cut_apply = True,
      xi_cut_value = 5. * 0.0020627,
      xi_cut_mean = +8.012857e-5,

      th_y_cut_apply = False
    )
  )

ctpps_2017.toModify(ctppsProtons, apply2017Settings)

def apply2018Settings(ctppsProtons):
  ctppsProtons.association_cuts_45.xi_cut_value = 0.013
  ctppsProtons.association_cuts_45.th_y_cut_apply = True
  ctppsProtons.association_cuts_45.th_y_cut_value = 30E-6

  ctppsProtons.association_cuts_56.xi_cut_value = 0.013
  ctppsProtons.association_cuts_56.th_y_cut_apply = True
  ctppsProtons.association_cuts_56.th_y_cut_value = 20E-6

  # update for re-miniAOD
  run2_miniAOD_UL.toModify(ctppsProtons,
    association_cuts_45 = dict(
      x_cut_apply = True,
      x_cut_value = 4. * 0.16008188,
      x_cut_mean = -0.065194856,

      y_cut_apply = True,
      y_cut_value = 4. * 0.1407986,
      y_cut_mean = +0.10973631,

      xi_cut_apply = True,
      xi_cut_value = 4. * 0.0012403586,
      xi_cut_mean = +3.113062e-5,

      th_y_cut_apply = False
    ),

    association_cuts_56 = dict(
      x_cut_apply = True,
      x_cut_value = 5. * 0.18126434,
      x_cut_mean = +0.073016431,

      y_cut_apply = True,
      y_cut_value = 5. * 0.14990802,
      y_cut_mean = +0.064261029,

      xi_cut_apply = True,
      xi_cut_value = 5. * 0.002046409,
      xi_cut_mean = -1.1852528e-5,

      th_y_cut_apply = False
    )
  )

ctpps_2018.toModify(ctppsProtons, apply2018Settings)
