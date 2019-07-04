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

def ApplyDefaultSettings(ctppsProtons):
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

ctpps_2016.toModify(ctppsProtons, ApplyDefaultSettings) # applied for all Run2 years (2016, 2017 and 2018)

def Apply2017Settings(ctppsProtons):
  ctppsProtons.association_cuts_45.xi_cut_value = 0.010
  ctppsProtons.association_cuts_56.xi_cut_value = 0.015

ctpps_2017.toModify(ctppsProtons, Apply2017Settings)

def Apply2018Settings(ctppsProtons):
  ctppsProtons.association_cuts_45.xi_cut_value = 0.013
  ctppsProtons.association_cuts_45.th_y_cut_apply = True
  ctppsProtons.association_cuts_45.th_y_cut_value = 30E-6

  ctppsProtons.association_cuts_56.xi_cut_value = 0.013
  ctppsProtons.association_cuts_56.th_y_cut_apply = True
  ctppsProtons.association_cuts_56.th_y_cut_value = 20E-6

ctpps_2018.toModify(ctppsProtons, Apply2018Settings)
