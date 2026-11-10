import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTSingleInOutLUT import COS_PHI_LUT as _COS_PHI_LUT
from L1Trigger.Phase2L1GT.l1tGTSingleInOutLUT import COSH_ETA_LUT as _COSH_ETA_LUT
from L1Trigger.Phase2L1GT.l1tGTSingleInOutLUT import COSH_ETA_LUT_2 as _COSH_ETA_LUT_2

# cos/cosh LUTs of the correlational and 3-body cuts as top-level PSets. The
# l1tGT{Double,Triple,Quad}ObjectCond modules reference them with refToPSet_, so a configuration
# holds a single copy of each LUT instead of one per condition. Any configuration using these
# conditions must load this file. The firmware LUT files are written from the same objects
# (SingleInOutLUT.export, see test/gt_firmware_evaluation.py).
l1tGTCoshEtaLUT = _COSH_ETA_LUT.config()
l1tGTCoshEtaLUT2 = _COSH_ETA_LUT_2.config()
l1tGTCosPhiLUT = _COS_PHI_LUT.config()
