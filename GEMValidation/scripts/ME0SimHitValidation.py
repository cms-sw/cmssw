import sys

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
def me0SimHitOccupancyXY():
  draw_occ(targetDir, "sh_me0_xy_rm1_l1" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rm1,l1,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rm1_l2" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rm1,l2,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rm1_l3" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region-1, layer3;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rm1,l3,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rm1_l4" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region-1, layer4;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rm1,l4,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rm1_l5" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region-1, layer5;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rm1,l5,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rm1_l6" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region-1, layer6;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rm1,l6,sel), "COLZ")
  
  draw_occ(targetDir, "sh_me0_xy_rp1_l1" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rp1,l1,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rp1_l2" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rp1,l2,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rp1_l3" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region1, layer3;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rp1,l3,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rp1_l4" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region1, layer4;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rp1,l4,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rp1_l5" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region1, layer5;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rp1,l5,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_xy_rp1_l6" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region1, layer6;globalX [cm];globalY [cm]", 
           "h_", "(120,-280,280,120,-280,280)", "globalX:globalY", AND(rp1,l6,sel), "COLZ")

#_______________________________________________________________________________
def me0SimHitOccupancyRZ():
  draw_occ(targetDir, "sh_me0_zr_rm1" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region-1;globalZ [cm];globalR [cm]", 
           "h_", "(80,-555,-515,120,20,280)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rm1,sel), "COLZ")
  draw_occ(targetDir, "sh_me0_zr_rp1" + suff, ext, treeME0SimHits, pre + " SimHit occupancy: region1;globalZ [cm];globalR [cm]", 
           "h_", "(80,515,555,120,20,280)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rp1,sel), "COLZ")

#_______________________________________________________________________________
def me0SimHitTOF():
  draw_1D(targetDir, "sh_me0_tof_rm1_l1" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region-1, layer1;Time of flight [ns];entries", 
          "h_", "(40,15,19)", "timeOfFlight", AND(rm1,l1,sel))
  draw_1D(targetDir, "sh_me0_tof_rm1_l2" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region-1, layer2;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l2,sel))
  draw_1D(targetDir, "sh_me0_tof_rm1_l3" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region-1, layer3;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l3,sel))
  draw_1D(targetDir, "sh_me0_tof_rm1_l4" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region-1, layer4;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l4,sel))
  draw_1D(targetDir, "sh_me0_tof_rm1_l5" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region-1, layer5;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l5,sel))
  draw_1D(targetDir, "sh_me0_tof_rm1_l6" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region-1, layer6;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l6,sel))
  
  draw_1D(targetDir, "sh_me0_tof_rp1_l1" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region1, layer1;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l1,sel))
  draw_1D(targetDir, "sh_me0_tof_rp1_l2" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region1, layer2;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l2,sel))
  draw_1D(targetDir, "sh_me0_tof_rp1_l3" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region1, layer3;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l3,sel))
  draw_1D(targetDir, "sh_me0_tof_rp1_l4" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region1, layer4;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l4,sel))
  draw_1D(targetDir, "sh_me0_tof_rp1_l5" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region1, layer5;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l5,sel))
  draw_1D(targetDir, "sh_me0_tof_rp1_l6" + suff, ext, treeME0SimHits, pre + " SimHit TOF: region1, layer6;Time of flight [ns];entries", 
          "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l6,sel))
  
#_______________________________________________________________________________
def me0SimTrackToSimHitMatchingLX(plotter):
  pass

#_______________________________________________________________________________
def me0SimTrackToSimHitMatchingLY(plotter):
  pass

#_______________________________________________________________________________
def me0SimTrackToSimHitMatchingEta(plotter): 
  pass

#_______________________________________________________________________________
def me0SimTrackToSimHitMatchingPhi(plotter):
  pass
