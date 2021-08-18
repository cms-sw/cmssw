import FWCore.ParameterSet.Config as cms

# import default alignment settings
from CalibPPS.ESProducers.ctppsAlignment_cff import *

# import default optics settings
from CalibPPS.ESProducers.ctppsOpticalFunctions_cff import *

# import and adjust proton-reconstructions settings
from RecoPPS.ProtonReconstruction.ctppsProtons_cfi import *
ctppsProtons.lhcInfoLabel = ctppsLHCInfoLabel

from Configuration.Eras.Modifier_ctpps_cff import ctpps
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018

def applyDefaultSettings(ctppsProtons):
  ctppsProtons.association_cuts_45.x_cut_apply = False
  ctppsProtons.association_cuts_45.y_cut_apply = False
  ctppsProtons.association_cuts_45.xi_cut_apply = True
  ctppsProtons.association_cuts_45.xi_cut_value = 0.010
  ctppsProtons.association_cuts_45.th_y_cut_apply = False
  ctppsProtons.association_cuts_45.ti_tr_min = -1.5
  ctppsProtons.association_cuts_45.ti_tr_max = 2.0

  ctppsProtons.association_cuts_56.x_cut_apply = False
  ctppsProtons.association_cuts_56.y_cut_apply = False
  ctppsProtons.association_cuts_56.xi_cut_apply = True
  ctppsProtons.association_cuts_56.xi_cut_value = 0.015
  ctppsProtons.association_cuts_56.th_y_cut_apply = False
  ctppsProtons.association_cuts_56.ti_tr_min = -1.5
  ctppsProtons.association_cuts_56.ti_tr_max = 2.0

  ctppsProtons.pixelDiscardBXShiftedTracks = True
  ctppsProtons.default_time = -999.

ctpps.toModify(ctppsProtons, applyDefaultSettings)

def apply2017Settings(ctppsProtons):
  ctppsProtons.association_cuts_45.x_cut_apply = False
  ctppsProtons.association_cuts_45.y_cut_apply = False

  ctppsProtons.association_cuts_45.xi_cut_apply = True
  ctppsProtons.association_cuts_45.xi_cut_value = 5. * 0.00121
  ctppsProtons.association_cuts_45.xi_cut_mean = +6.0695e-5

  ctppsProtons.association_cuts_45.th_y_cut_apply = False

  ctppsProtons.association_cuts_56.x_cut_apply = False

  ctppsProtons.association_cuts_56.y_cut_apply = True
  ctppsProtons.association_cuts_56.y_cut_value = 5. * 0.14777
  ctppsProtons.association_cuts_56.y_cut_mean = -0.022612

  ctppsProtons.association_cuts_56.xi_cut_apply = True
  ctppsProtons.association_cuts_56.xi_cut_value = 5. * 0.0020627
  ctppsProtons.association_cuts_56.xi_cut_mean = +8.012857e-5

  ctppsProtons.association_cuts_56.th_y_cut_apply = False

ctpps_2017.toModify(ctppsProtons, apply2017Settings)

def apply2018Settings(ctppsProtons):
  ctppsProtons.association_cuts_45.x_cut_apply = True
  ctppsProtons.association_cuts_45.x_cut_value = 4. * 0.16008188
  ctppsProtons.association_cuts_45.x_cut_mean = -0.065194856

  ctppsProtons.association_cuts_45.y_cut_apply = True
  ctppsProtons.association_cuts_45.y_cut_value = 4. * 0.1407986
  ctppsProtons.association_cuts_45.y_cut_mean = +0.10973631

  ctppsProtons.association_cuts_45.xi_cut_apply = True
  ctppsProtons.association_cuts_45.xi_cut_value = 4. * 0.0012403586
  ctppsProtons.association_cuts_45.xi_cut_mean = +3.113062e-5

  ctppsProtons.association_cuts_45.th_y_cut_apply = False

  ctppsProtons.association_cuts_56.x_cut_apply = True
  ctppsProtons.association_cuts_56.x_cut_value = 5. * 0.18126434
  ctppsProtons.association_cuts_56.x_cut_mean = +0.073016431

  ctppsProtons.association_cuts_56.y_cut_apply = True
  ctppsProtons.association_cuts_56.y_cut_value = 5. * 0.14990802
  ctppsProtons.association_cuts_56.y_cut_mean = +0.064261029

  ctppsProtons.association_cuts_56.xi_cut_apply = True
  ctppsProtons.association_cuts_56.xi_cut_value = 5. * 0.002046409
  ctppsProtons.association_cuts_56.xi_cut_mean = -1.1852528e-5

  ctppsProtons.association_cuts_56.th_y_cut_apply = False

ctpps_2018.toModify(ctppsProtons, apply2018Settings)
