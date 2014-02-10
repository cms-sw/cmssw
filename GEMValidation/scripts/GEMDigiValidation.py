from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)


#_______________________________________________________________________________
def gemGEMDigiOccupancyXY(plotter):
  ## station 1
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st1_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st1_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l2), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st1_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st1_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l2), "COLZ") 

  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st1_l1_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station1, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l1,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st1_l2_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station1, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l2,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st1_l1_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station1, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l1,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st1_l2_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station1, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l2,odd), "COLZ")

  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st1_l1_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station1, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l1,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st1_l2_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station1, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l2,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st1_l1_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station1, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l1,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st1_l2_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station1, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l2,even), "COLZ")

  ## station 2
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st2_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station2, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st2_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station2, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l2), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st2_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station2, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st2_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station2, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l2), "COLZ") 

  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st2_l1_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station2, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l1,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st2_l2_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station2, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l2,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st2_l1_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station2, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l1,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st2_l2_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station2, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l2,odd), "COLZ")

  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st2_l1_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station2, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l1,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st2_l2_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station2, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l2,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st2_l1_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station2, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l1,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st2_l2_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station2, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l2,even), "COLZ")

  ## station 3
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st3_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station3, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st3_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station3, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l2), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st3_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station3, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st3_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station3, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l2), "COLZ") 

  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st3_l1_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station3, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l1,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st3_l2_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station3, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l2,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st3_l1_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station3, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l1,odd), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st3_l2_odd", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station3, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l2,odd), "COLZ")

  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st3_l1_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station3, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l1,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rm1_st3_l2_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1, station3, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l2,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st3_l1_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station3, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l1,even), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_xy_rp1_st3_l2_even", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1, station3, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l2,even), "COLZ")
  

#_______________________________________________________________________________
def gemGEMDigiOccupancyStripPhi(plotter):
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rm1_st1_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1 station1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE11/2,plotter.nstripsGE11), "strip:g_phi", AND(rm1,st1,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rm1_st1_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1 station1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE11/2,plotter.nstripsGE11), "strip:g_phi", AND(rm1,st1,l2), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rp1_st1_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1 station1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE11/2,plotter.nstripsGE11), "strip:g_phi", AND(rp1,st1,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rp1_st1_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1 station1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE11/2,plotter.nstripsGE11), "strip:g_phi", AND(rp1,st1,l2), "COLZ")
 
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rm1_st2_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1 station2 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rm1,st2,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rm1_st2_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1 station2 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rm1,st2,l2), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rp1_st2_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1 station2 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rp1,st2,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rp1_st2_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1 station2 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rp1,st2,l2), "COLZ")

  draw_occ(plotter.targetDir, "strip_dg_phistrip_rm1_st3_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1 station3 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rm1,st3,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rm1_st3_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1 station3 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rm1,st3,l2), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rp1_st3_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1 station3 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rp1,st3,l1), "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_phistrip_rp1_st3_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1 station3 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(plotter.nstripsGE21/2,plotter.nstripsGE21), "strip:g_phi", AND(rp1,st3,l2), "COLZ")


#_______________________________________________________________________________
def gemGEMDigiOccupancyStrip(plotter):
  draw_1D(plotter.targetDir, "strip_dg_rm1_st1_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region-1 station1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rm1,st1,l1))
  draw_1D(plotter.targetDir, "strip_dg_rm1_st1_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region-1 station1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rm1,st1,l2))
  draw_1D(plotter.targetDir, "strip_dg_rp1_st1_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region1 station1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rp1,st1,l1))
  draw_1D(plotter.targetDir, "strip_dg_rp1_st1_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region1 station1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rp1,st1,l2))
  
  draw_1D(plotter.targetDir, "strip_dg_rm1_st2_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region-1 station2 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st2,l1))
  draw_1D(plotter.targetDir, "strip_dg_rm1_st2_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region-1 station2 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st2,l2))
  draw_1D(plotter.targetDir, "strip_dg_rp1_st2_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region1 station2 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st2,l1))
  draw_1D(plotter.targetDir, "strip_dg_rp1_st2_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region1 station2 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st2,l2))

  draw_1D(plotter.targetDir, "strip_dg_rm1_st3_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region-1 station3 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st3,l1))
  draw_1D(plotter.targetDir, "strip_dg_rm1_st3_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region-1 station3 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st3,l2))
  draw_1D(plotter.targetDir, "strip_dg_rp1_st3_l1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region1 station3 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st3,l1))
  draw_1D(plotter.targetDir, "strip_dg_rp1_st3_l2", plotter.ext, plotter.treeGEMDigis, "Digi occupancy per strip number, region1 station3 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st3,l2))
    

#_______________________________________________________________________________
def gemGEMDigiBX(plotter):
  draw_bx(plotter.targetDir, "strip_digi_bx_rm1_l1", plotter.ext, plotter.treeGEMDigis, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l1))
  draw_bx(plotter.targetDir, "strip_digi_bx_rm1_l2", plotter.ext, plotter.treeGEMDigis, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l2))
  draw_bx(plotter.targetDir, "strip_digi_bx_rp1_l1", plotter.ext, plotter.treeGEMDigis, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l1))
  draw_bx(plotter.targetDir, "strip_digi_bx_rp1_l2", plotter.ext, plotter.treeGEMDigis, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l2))


#_______________________________________________________________________________
def gemGEMDigiOccupancyRZ(plotter):
  draw_occ(plotter.targetDir, "strip_dg_zr_rm1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region-1; globalZ [cm]; globalR [cm]", 
           "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(plotter.targetDir, "strip_dg_zr_rp1", plotter.ext, plotter.treeGEMDigis, "Digi occupancy: region1; globalZ [cm]; globalR [cm]", 
           "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")


#_______________________________________________________________________________
def gemGEMPadOccupancyXY(plotter):
  draw_occ(plotter.targetDir, "pad_dg_xy_rm1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_xy_rm1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,l2), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_xy_rp1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_xy_rp1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,l2), "COLZ")


#_______________________________________________________________________________
def gemGEMPadOccupancyPadPhi(plotter):
  draw_occ(plotter.targetDir, "pad_dg_phipad_rm1_st1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1 station1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE11/2.) + ",0, %f" %(plotter.npadsGE11) + ")", "pad:g_phi", AND(rm1,st1,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rm1_st1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1 station1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE11/2.) + ",0, %f" %(plotter.npadsGE11) + ")", "pad:g_phi", AND(rm1,st1,l2), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rp1_st1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1 station1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE11/2.) + ",0, %f" %(plotter.npadsGE11) + ")", "pad:g_phi", AND(rp1,st1,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rp1_st1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1 station1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE11/2.) + ",0, %f" %(plotter.npadsGE11) + ")", "pad:g_phi", AND(rp1,st1,l2), "COLZ")

  draw_occ(plotter.targetDir, "pad_dg_phipad_rm1_st2_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1 station2 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rm1,st2,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rm1_st2_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1 station2 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rm1,st2,l2), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rp1_st2_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1 station2 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rp1,st2,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rp1_st2_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1 station2 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rp1,st2,l2), "COLZ")

  draw_occ(plotter.targetDir, "pad_dg_phipad_rm1_st3_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1 station3 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rm1,st3,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rm1_st3_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1 station3 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rm1,st3,l2), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rp1_st3_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1 station3 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rp1,st3,l1), "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_phipad_rp1_st3_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1 station3 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE21/2.) + ",0, %f" %(plotter.npadsGE21) + ")", "pad:g_phi", AND(rp1,st3,l2), "COLZ")


#_______________________________________________________________________________
def gemGEMPadOccupancyPad(plotter):
  draw_1D(plotter.targetDir, "pad_dg_rm1_st1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region-1 station1 layer1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE11) + ",0.5, %f" %(plotter.npadsGE11 + 0.5) + ")", "pad", AND(rm1,st1,l1))
  draw_1D(plotter.targetDir, "pad_dg_rm1_st1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region-1 station1 layer2;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE11) + ",0.5, %f" %(plotter.npadsGE11 + 0.5) + ")", "pad", AND(rm1,st1,l2))
  draw_1D(plotter.targetDir, "pad_dg_rp1_st1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region1 station1 layer1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE11) + ",0.5, %f" %(plotter.npadsGE11 + 0.5) + ")", "pad", AND(rp1,st1,l1))
  draw_1D(plotter.targetDir, "pad_dg_rp1_st1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region1 station1 layer2;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE11) + ",0.5, %f" %(plotter.npadsGE11 + 0.5) + ")", "pad", AND(rp1,st1,l2))

  draw_1D(plotter.targetDir, "pad_dg_rm1_st2_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region-1 station2 layer1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rm1,st2,l1))
  draw_1D(plotter.targetDir, "pad_dg_rm1_st2_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region-1 station2 layer2;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rm1,st2,l2))
  draw_1D(plotter.targetDir, "pad_dg_rp1_st2_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region1 station2 layer1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rp1,st2,l1))
  draw_1D(plotter.targetDir, "pad_dg_rp1_st2_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region1 station2 layer2;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rp1,st2,l2))

  draw_1D(plotter.targetDir, "pad_dg_rm1_st3_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region-1 station3 layer1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rm1,st3,l1))
  draw_1D(plotter.targetDir, "pad_dg_rm1_st3_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region-1 station3 layer2;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rm1,st3,l2))
  draw_1D(plotter.targetDir, "pad_dg_rp1_st3_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region1 station3 layer1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rp1,st3,l1))
  draw_1D(plotter.targetDir, "pad_dg_rp1_st3_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Digi occupancy per pad number, region1 station3 layer2;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE21) + ",0.5, %f" %(plotter.npadsGE21 + 0.5) + ")", "pad", AND(rp1,st3,l2))


#_______________________________________________________________________________
def gemGEMPadBX(plotter):
  draw_bx(plotter.targetDir, "pad_dg_bx_rm1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l1))
  draw_bx(plotter.targetDir, "pad_dg_bx_rm1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l2))
  draw_bx(plotter.targetDir, "pad_dg_bx_rp1_l1", plotter.ext, plotter.treeGEMCSPadDigis, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l1))
  draw_bx(plotter.targetDir, "pad_dg_bx_rp1_l2", plotter.ext, plotter.treeGEMCSPadDigis, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l2))

    
#_______________________________________________________________________________
def gemGEMCoPadOccupancyCoPadPhi(plotter):
  draw_occ(plotter.targetDir, "copad_dg_phipad_rm1_l1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Pad occupancy: region-1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE11/2.) + ",0, %f" %(plotter.npadsGE11) + ")", "pad:g_phi", rm1, "COLZ")
  draw_occ(plotter.targetDir, "copad_dg_phipad_rp1_l1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Pad occupancy: region1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(plotter.npadsGE11/2.) + ",0, %f" %(plotter.npadsGE11) + ")", "pad:g_phi", rp1, "COLZ")

    
#_______________________________________________________________________________
def gemGEMCoPadOccupancyCoPad(plotter):
  draw_1D(plotter.targetDir, "copad_dg_rm1_l1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Digi occupancy per pad number, region-1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE11) + ",0.5, %f" %(plotter.npadsGE11 + 0.5) +  ")", "pad", rm1)
  draw_1D(plotter.targetDir, "copad_dg_rp1_l1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Digi occupancy per pad number, region1;pad number;entries", 
	  "h_", "( %f" %(plotter.npadsGE11) + ",0.5, %f" %(plotter.npadsGE11 + 0.5) +  ")", "pad", rp1)


#_______________________________________________________________________________
def gemGEMCoPadOccupancyXY(plotter):
  draw_occ(plotter.targetDir, "copad_dg_xy_rm1_l1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Pad occupancy: region-1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rm1, "COLZ")
  draw_occ(plotter.targetDir, "copad_dg_xy_rp1_l1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Pad occupancy: region1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rp1, "COLZ")


#_______________________________________________________________________________
def gemGEMCoPadBX(plotter):
  draw_bx(plotter.targetDir, "copad_dg_bx_rm1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Bunch crossing: region-1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rm1)
  draw_bx(plotter.targetDir, "copad_dg_bx_rp1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Bunch crossing: region1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rp1)


#_______________________________________________________________________________
def gemGEMPadOccupancyRZ(plotter):
  draw_occ(plotter.targetDir, "pad_dg_zr_rm1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(plotter.targetDir, "pad_dg_zr_rp1", plotter.ext, plotter.treeGEMCSPadDigis, "Pad occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")


#_______________________________________________________________________________
def gemGEMCoPadOccupancyRZ(plotter):
  draw_occ(plotter.targetDir, "copad_dg_zr_rm1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Pad occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(plotter.targetDir, "copad_dg_zr_rp1", plotter.ext, plotter.treeGEMCSCoPadDigis, "Pad occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")

#_______________________________________________________________________________
def simTrackDigiMatchingEta(plotter):
  ## digis
  draw_geff(plotter.targetDir, "eff_eta_track_dg_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, ok_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_dg_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, ok_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_dg_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_dg_gem_l1and2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, AND(ok_gL2dg,ok_gL1dg), "P", kBlue)

  ## digis with matched simhits
  draw_geff(plotter.targetDir, "eff_eta_track_dg_sh_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_dg_sh_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_dg_sh_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", OR(ok_gL2sh,ok_gL1sh),
            OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_dg_sh_gem_l1and2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", AND(ok_gL2sh,ok_gL1sh),
            AND(ok_gL2dg,ok_gL1dg), "P", kBlue)

#_______________________________________________________________________________
def simTrackDigiMatchingPhi(plotter):
  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l1and2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, AND(ok_gL2dg,ok_gL1dg), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh), ok_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL2sh), ok_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,OR(ok_gL1sh,ok_gL2sh)),
            OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_dg_gem_l1and2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh,ok_gL2sh),
            AND(ok_gL2dg,ok_gL1dg), "P", kBlue)


#_______________________________________________________________________________
def simTrackDigiMatchingLX(plotter):
  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l1_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, ok_trk_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, ok_trk_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l1or2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l1and2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l1_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, ok_trk_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, ok_trk_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l1or2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_dg_gem_l1and2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)

#_______________________________________________________________________________
def simTrackDigiMatchingLY(plotter):
  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l1_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", nocut, ok_trk_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", nocut, ok_trk_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l1or2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", nocut, OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l1and2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", nocut, AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l1_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", nocut, ok_trk_gL1dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", nocut, ok_trk_gL2dg, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l1or2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", nocut, OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_dg_gem_l1and2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", nocut, AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)

#_______________________________________________________________________________
def simTrackPadMatchingEta(plotter):
  draw_geff(plotter.targetDir, "eff_eta_track_pad_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, ok_gL1pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_pad_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, ok_gL2pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_pad_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, OR(ok_gL2pad,ok_gL1pad), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_eta_track_pad_sh_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_pad_sh_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_pad_sh_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", OR(ok_gL1sh,ok_gL2sh),
            OR(ok_gL2pad,ok_gL1pad), "P", kBlue)


#_______________________________________________________________________________
def simTrackPadMatchingPhi(plotter):
  draw_geff(plotter.targetDir, "eff_phi_track_pad_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_pad_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_pad_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta,
            OR(ok_gL2pad,ok_gL1pad), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_phi_track_pad_sh_gem_l1", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh), ok_gL1pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_pad_sh_gem_l2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL2sh), ok_gL2pad, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_pad_sh_gem_l1or2", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,OR(ok_gL1sh,ok_gL2sh)),
            OR(ok_gL2pad,ok_gL1pad), "P", kBlue)


#_______________________________________________________________________________
def simTrackPadMatchingLX(plotter):
    pass


#_______________________________________________________________________________
def simTrackPadMatchingLY(plotter):
    pass


#_______________________________________________________________________________
def simTrackCoPadMatchingEta(plotter):
  draw_geff(plotter.targetDir, "eff_eta_track_copad_gem", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, AND(ok_gL1pad,ok_gL2pad), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_eta_track_copad_sh_gem", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", AND(ok_gL1sh,ok_gL2sh),
            AND(ok_gL1pad,ok_gL2pad), "P", kBlue)

#_______________________________________________________________________________
def simTrackCoPadMatchingPhi(plotter):
  draw_geff(plotter.targetDir, "eff_phi_track_copad_gem", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta,AND(ok_gL1pad,ok_gL2pad), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_phi_track_copad_sh_gem", plotter.ext, plotter.treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh,ok_gL2sh),
            AND(ok_gL2pad,ok_gL1pad), "P", kBlue)

#_______________________________________________________________________________
def simTrackCoPadMatchingLX(plotter):
    pass


#_______________________________________________________________________________
def simTrackCoPadMatchingLY(plotter):
    pass
