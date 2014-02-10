import sys

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

def gemGEMDigiOccupancyXY():
  pass

def gemGEMCSCPadDigiOccupancyXY():
  pass

def gemGEMCSCCoPadDigiOccupancyXY():
  pass

def gemGEMCSCCoPadDigiOccupancyXY():
  pass

def gemGEMDigiOccupancyRZ():
  pass

def gemGEMCSCPadDigiOccupancyRZ():
  pass

def gemGEMCSCCoPadDigiOccupancyRZ():
  pass

def gemGEMCSCCoPadDigiOccupancyRZ():
  pass


if __name__ == "__main__":  

  inputFile = str(sys.argv[1])
  if len(inputFile) < 3:
      inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/gem_digi_ana.root'
      inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/fixStripsPads/test2/CMSSW_6_2_0_SLHC5/src/gem_digi_ana.root'
      inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/testForGeometry/CMSSW_6_2_0_SLHC7/src/gem_digi_ana.root'
  targetDir = './'
  ## extension for figures - add more?
  ext = ".png"
  
  ## strips and pads'
  nstripsGE11 = 384
  nstripsGE21 = 768
  npadsGE11 = 96
  npadsGE21 = 192

  ## Trees
  analyzer = "MuonDigiAnalyzer"
  simHits = "GEMSimHits"
  digis = "GEMDigiTree"
  pads = "GEMCSCPadDigiTree"
  copads = "GEMCSCCoPadDigiTree"
  simTracks = "TrackTree"

  ## Style
  gStyle.SetStatStyle(0);

    ## input
  file = TFile.Open(inputFile)
  if not file:
    sys.exit('Input ROOT file %s is missing.' %(inputFile))

  dirAna = file.Get(analyzer)
  if not dirAna:
    sys.exit('Directory %s does not exist.' %(dirAna))
    
  treeDigis = dirAna.Get(digis)
  if not treeDigis:
    sys.exit('Tree %s does not exist.' %(treeDigis))

  ## occupancy plots
  draw_occ(targetDir, "strip_dg_xy_rm1_st1_l1", ext, treeDigis, "Digi occupancy: region-1, station1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rm1_st1_l2", ext, treeDigis, "Digi occupancy: region-1, station1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l2), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st1_l1", ext, treeDigis, "Digi occupancy: region1, station1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st1_l2", ext, treeDigis, "Digi occupancy: region1, station1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l2), "COLZ") 

  draw_occ(targetDir, "strip_dg_xy_rm1_st1_l1_odd", ext, treeDigis, "Digi occupancy: region-1, station1, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l1,odd), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rm1_st1_l2_odd", ext, treeDigis, "Digi occupancy: region-1, station1, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l2,odd), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st1_l1_odd", ext, treeDigis, "Digi occupancy: region1, station1, layer1, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l1,odd), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st1_l2_odd", ext, treeDigis, "Digi occupancy: region1, station1, layer2, Odd; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l2,odd), "COLZ")

  draw_occ(targetDir, "strip_dg_xy_rm1_st1_l1_even", ext, treeDigis, "Digi occupancy: region-1, station1, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l1,even), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rm1_st1_l2_even", ext, treeDigis, "Digi occupancy: region-1, station1, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st1,l2,even), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st1_l1_even", ext, treeDigis, "Digi occupancy: region1, station1, layer1, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l1,even), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st1_l2_even", ext, treeDigis, "Digi occupancy: region1, station1, layer2, Even; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st1,l2,even), "COLZ")

  ## station 2

  draw_occ(targetDir, "strip_dg_xy_rm1_st2_l1", ext, treeDigis, "Digi occupancy: region-1, station2, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rm1_st2_l2", ext, treeDigis, "Digi occupancy: region-1, station2, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st2,l2), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st2_l1", ext, treeDigis, "Digi occupancy: region1, station2, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st2_l2", ext, treeDigis, "Digi occupancy: region1, station2, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st2,l2), "COLZ") 

  draw_occ(targetDir, "strip_dg_xy_rm1_st3_l1", ext, treeDigis, "Digi occupancy: region-1, station3, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rm1_st3_l2", ext, treeDigis, "Digi occupancy: region-1, station3, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,st3,l2), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st3_l1", ext, treeDigis, "Digi occupancy: region1, station3, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_st3_l2", ext, treeDigis, "Digi occupancy: region1, station3, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,st3,l2), "COLZ") 

  draw_occ(targetDir, "strip_dg_zr_rm1", ext, treeDigis, "Digi occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(targetDir, "strip_dg_zr_rp1", ext, treeDigis, "Digi occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")

  draw_occ(targetDir, "strip_dg_phistrip_rm1_st1_l1", ext, treeDigis, "Digi occupancy: region-1 station1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE11/2,nstripsGE11), "strip:g_phi", AND(rm1,st1,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rm1_st1_l2", ext, treeDigis, "Digi occupancy: region-1 station1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE11/2,nstripsGE11), "strip:g_phi", AND(rm1,st1,l2), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_st1_l1", ext, treeDigis, "Digi occupancy: region1 station1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE11/2,nstripsGE11), "strip:g_phi", AND(rp1,st1,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_st1_l2", ext, treeDigis, "Digi occupancy: region1 station1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE11/2,nstripsGE11), "strip:g_phi", AND(rp1,st1,l2), "COLZ")
 
  draw_occ(targetDir, "strip_dg_phistrip_rm1_st2_l1", ext, treeDigis, "Digi occupancy: region-1 station2 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rm1,st2,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rm1_st2_l2", ext, treeDigis, "Digi occupancy: region-1 station2 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rm1,st2,l2), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_st2_l1", ext, treeDigis, "Digi occupancy: region1 station2 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rp1,st2,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_st2_l2", ext, treeDigis, "Digi occupancy: region1 station2 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rp1,st2,l2), "COLZ")

  draw_occ(targetDir, "strip_dg_phistrip_rm1_st3_l1", ext, treeDigis, "Digi occupancy: region-1 station3 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rm1,st3,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rm1_st3_l2", ext, treeDigis, "Digi occupancy: region-1 station3 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rm1,st3,l2), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_st3_l1", ext, treeDigis, "Digi occupancy: region1 station3 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rp1,st3,l1), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_st3_l2", ext, treeDigis, "Digi occupancy: region1 station3 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,%d,0,%d)"%(nstripsGE21/2,nstripsGE21), "strip:g_phi", AND(rp1,st3,l2), "COLZ")

  draw_1D(targetDir, "strip_dg_rm1_st1_l1", ext, treeDigis, "Digi occupancy per strip number, region-1 station1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rm1,st1,l1))
  draw_1D(targetDir, "strip_dg_rm1_st1_l2", ext, treeDigis, "Digi occupancy per strip number, region-1 station1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rm1,st1,l2))
  draw_1D(targetDir, "strip_dg_rp1_st1_l1", ext, treeDigis, "Digi occupancy per strip number, region1 station1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rp1,st1,l1))
  draw_1D(targetDir, "strip_dg_rp1_st1_l2", ext, treeDigis, "Digi occupancy per strip number, region1 station1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", AND(rp1,st1,l2))
  
  draw_1D(targetDir, "strip_dg_rm1_st2_l1", ext, treeDigis, "Digi occupancy per strip number, region-1 station2 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st2,l1))
  draw_1D(targetDir, "strip_dg_rm1_st2_l2", ext, treeDigis, "Digi occupancy per strip number, region-1 station2 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st2,l2))
  draw_1D(targetDir, "strip_dg_rp1_st2_l1", ext, treeDigis, "Digi occupancy per strip number, region1 station2 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st2,l1))
  draw_1D(targetDir, "strip_dg_rp1_st2_l2", ext, treeDigis, "Digi occupancy per strip number, region1 station2 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st2,l2))

  draw_1D(targetDir, "strip_dg_rm1_st3_l1", ext, treeDigis, "Digi occupancy per strip number, region-1 station3 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st3,l1))
  draw_1D(targetDir, "strip_dg_rm1_st3_l2", ext, treeDigis, "Digi occupancy per strip number, region-1 station3 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rm1,st3,l2))
  draw_1D(targetDir, "strip_dg_rp1_st3_l1", ext, treeDigis, "Digi occupancy per strip number, region1 station3 layer1;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st3,l1))
  draw_1D(targetDir, "strip_dg_rp1_st3_l2", ext, treeDigis, "Digi occupancy per strip number, region1 station3 layer2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "strip", AND(rp1,st3,l2))

  ## Bunch crossing plots
  draw_bx(targetDir, "strip_digi_bx_rm1_l1", ext, treeDigis, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l1))
  draw_bx(targetDir, "strip_digi_bx_rm1_l2", ext, treeDigis, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l2))
  draw_bx(targetDir, "strip_digi_bx_rp1_l1", ext, treeDigis, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l1))
  draw_bx(targetDir, "strip_digi_bx_rp1_l2", ext, treeDigis, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l2))

  treePads = dirAna.Get(pads)
  if not treePads:
    sys.exit('Tree %s does not exist.' %(treePads))

  ## occupancy plots
  draw_occ(targetDir, "pad_dg_xy_rm1_l1", ext, treePads, "Pad occupancy: region-1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_xy_rm1_l2", ext, treePads, "Pad occupancy: region-1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rm1,l2), "COLZ")
  draw_occ(targetDir, "pad_dg_xy_rp1_l1", ext, treePads, "Pad occupancy: region1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_xy_rp1_l2", ext, treePads, "Pad occupancy: region1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", AND(rp1,l2), "COLZ")

  draw_occ(targetDir, "pad_dg_zr_rm1", ext, treePads, "Pad occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(targetDir, "pad_dg_zr_rp1", ext, treePads, "Pad occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")

  draw_occ(targetDir, "pad_dg_phipad_rm1_st1_l1", ext, treePads, "Pad occupancy: region-1 station1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE11/2.) + ",0, %f" %(npadsGE11) + ")", "pad:g_phi", AND(rm1,st1,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rm1_st1_l2", ext, treePads, "Pad occupancy: region-1 station1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE11/2.) + ",0, %f" %(npadsGE11) + ")", "pad:g_phi", AND(rm1,st1,l2), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_st1_l1", ext, treePads, "Pad occupancy: region1 station1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE11/2.) + ",0, %f" %(npadsGE11) + ")", "pad:g_phi", AND(rp1,st1,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_st1_l2", ext, treePads, "Pad occupancy: region1 station1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE11/2.) + ",0, %f" %(npadsGE11) + ")", "pad:g_phi", AND(rp1,st1,l2), "COLZ")

  draw_occ(targetDir, "pad_dg_phipad_rm1_st2_l1", ext, treePads, "Pad occupancy: region-1 station2 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rm1,st2,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rm1_st2_l2", ext, treePads, "Pad occupancy: region-1 station2 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rm1,st2,l2), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_st2_l1", ext, treePads, "Pad occupancy: region1 station2 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rp1,st2,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_st2_l2", ext, treePads, "Pad occupancy: region1 station2 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rp1,st2,l2), "COLZ")

  draw_occ(targetDir, "pad_dg_phipad_rm1_st3_l1", ext, treePads, "Pad occupancy: region-1 station3 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rm1,st3,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rm1_st3_l2", ext, treePads, "Pad occupancy: region-1 station3 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rm1,st3,l2), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_st3_l1", ext, treePads, "Pad occupancy: region1 station3 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rp1,st3,l1), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_st3_l2", ext, treePads, "Pad occupancy: region1 station3 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE21/2.) + ",0, %f" %(npadsGE21) + ")", "pad:g_phi", AND(rp1,st3,l2), "COLZ")

  draw_1D(targetDir, "pad_dg_rm1_st1_l1", ext, treePads, "Digi occupancy per pad number, region-1 station1 layer1;pad number;entries", 
	  "h_", "( %f" %(npadsGE11) + ",0.5, %f" %(npadsGE11 + 0.5) + ")", "pad", AND(rm1,st1,l1))
  draw_1D(targetDir, "pad_dg_rm1_st1_l2", ext, treePads, "Digi occupancy per pad number, region-1 station1 layer2;pad number;entries", 
	  "h_", "( %f" %(npadsGE11) + ",0.5, %f" %(npadsGE11 + 0.5) + ")", "pad", AND(rm1,st1,l2))
  draw_1D(targetDir, "pad_dg_rp1_st1_l1", ext, treePads, "Digi occupancy per pad number, region1 station1 layer1;pad number;entries", 
	  "h_", "( %f" %(npadsGE11) + ",0.5, %f" %(npadsGE11 + 0.5) + ")", "pad", AND(rp1,st1,l1))
  draw_1D(targetDir, "pad_dg_rp1_st1_l2", ext, treePads, "Digi occupancy per pad number, region1 station1 layer2;pad number;entries", 
	  "h_", "( %f" %(npadsGE11) + ",0.5, %f" %(npadsGE11 + 0.5) + ")", "pad", AND(rp1,st1,l2))

  draw_1D(targetDir, "pad_dg_rm1_st2_l1", ext, treePads, "Digi occupancy per pad number, region-1 station2 layer1;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rm1,st2,l1))
  draw_1D(targetDir, "pad_dg_rm1_st2_l2", ext, treePads, "Digi occupancy per pad number, region-1 station2 layer2;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rm1,st2,l2))
  draw_1D(targetDir, "pad_dg_rp1_st2_l1", ext, treePads, "Digi occupancy per pad number, region1 station2 layer1;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rp1,st2,l1))
  draw_1D(targetDir, "pad_dg_rp1_st2_l2", ext, treePads, "Digi occupancy per pad number, region1 station2 layer2;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rp1,st2,l2))

  draw_1D(targetDir, "pad_dg_rm1_st3_l1", ext, treePads, "Digi occupancy per pad number, region-1 station3 layer1;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rm1,st3,l1))
  draw_1D(targetDir, "pad_dg_rm1_st3_l2", ext, treePads, "Digi occupancy per pad number, region-1 station3 layer2;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rm1,st3,l2))
  draw_1D(targetDir, "pad_dg_rp1_st3_l1", ext, treePads, "Digi occupancy per pad number, region1 station3 layer1;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rp1,st3,l1))
  draw_1D(targetDir, "pad_dg_rp1_st3_l2", ext, treePads, "Digi occupancy per pad number, region1 station3 layer2;pad number;entries", 
	  "h_", "( %f" %(npadsGE21) + ",0.5, %f" %(npadsGE21 + 0.5) + ")", "pad", AND(rp1,st3,l2))

  ## Bunch crossing plots
  draw_bx(targetDir, "pad_dg_bx_rm1_l1", ext, treePads, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l1))
  draw_bx(targetDir, "pad_dg_bx_rm1_l2", ext, treePads, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rm1,l2))
  draw_bx(targetDir, "pad_dg_bx_rp1_l1", ext, treePads, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l1))
  draw_bx(targetDir, "pad_dg_bx_rp1_l2", ext, treePads, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", AND(rp1,l2))

  treeCoPads = dirAna.Get(copads)
  if not treeCoPads:
    sys.exit('Tree %s does not exist.' %(treeCoPads))

  ## occupancy plots
  draw_occ(targetDir, "copad_dg_xy_rm1_l1", ext, treeCoPads, "Pad occupancy: region-1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rm1, "COLZ")
  draw_occ(targetDir, "copad_dg_xy_rp1_l1", ext, treeCoPads, "Pad occupancy: region1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rp1, "COLZ")

  draw_occ(targetDir, "copad_dg_zr_rm1", ext, treeCoPads, "Pad occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(targetDir, "copad_dg_zr_rp1", ext, treeCoPads, "Pad occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")

  draw_occ(targetDir, "copad_dg_phipad_rm1_l1", ext, treeCoPads, "Pad occupancy: region-1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE11/2.) + ",0, %f" %(npadsGE11) + ")", "pad:g_phi", rm1, "COLZ")
  draw_occ(targetDir, "copad_dg_phipad_rp1_l1", ext, treeCoPads, "Pad occupancy: region1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npadsGE11/2.) + ",0, %f" %(npadsGE11) + ")", "pad:g_phi", rp1, "COLZ")
 
  draw_1D(targetDir, "copad_dg_rm1_l1", ext, treeCoPads, "Digi occupancy per pad number, region-1;pad number;entries", 
	  "h_", "( %f" %(npadsGE11) + ",0.5, %f" %(npadsGE11 + 0.5) +  ")", "pad", rm1)
  draw_1D(targetDir, "copad_dg_rp1_l1", ext, treeCoPads, "Digi occupancy per pad number, region1;pad number;entries", 
	  "h_", "( %f" %(npadsGE11) + ",0.5, %f" %(npadsGE11 + 0.5) +  ")", "pad", rp1)

  ## Bunch crossing plots
  draw_bx(targetDir, "copad_dg_bx_rm1", ext, treeCoPads, "Bunch crossing: region-1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rm1)
  draw_bx(targetDir, "copad_dg_bx_rp1", ext, treeCoPads, "Bunch crossing: region1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rp1)

  ## Tracks
  treeTracks = dirAna.Get(simTracks)
  if not treeTracks:
    sys.exit('Tree %s does not exist.' %(treeTracks))

  ## digis
  draw_geff(targetDir, "eff_eta_track_dg_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), AND(ok_gL2dg,ok_gL1dg), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_dg_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, AND(ok_gL2dg,ok_gL1dg), "P", kBlue)

  ## digis with matched simhits
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", OR(ok_gL2sh,ok_gL1sh),
            OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", AND(ok_gL2sh,ok_gL1sh),
            AND(ok_gL2dg,ok_gL1dg), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_dg_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh), ok_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL2sh), ok_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,OR(ok_gL1sh,ok_gL2sh)),
            OR(ok_gL2dg,ok_gL1dg), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh,ok_gL2sh),
            AND(ok_gL2dg,ok_gL1dg), "P", kBlue)

  ## pads
  draw_geff(targetDir, "eff_eta_track_pad_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), OR(ok_gL2pad,ok_gL1pad), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_pad_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta,
            OR(ok_gL2pad,ok_gL1pad), "P", kBlue)

  ## pads with matched simhits
  draw_geff(targetDir, "eff_eta_track_pad_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", OR(ok_gL1sh,ok_gL2sh),
            OR(ok_gL2pad,ok_gL1pad), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_pad_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh), ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL2sh), ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,OR(ok_gL1sh,ok_gL2sh)),
            OR(ok_gL2pad,ok_gL1pad), "P", kBlue)

  ## copads
  draw_geff(targetDir, "eff_eta_track_copad_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), AND(ok_gL1pad,ok_gL2pad), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_copad_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta,AND(ok_gL1pad,ok_gL2pad), "P", kBlue)

  ## copads with matched simhits
  draw_geff(targetDir, "eff_eta_track_copad_sh_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", AND(ok_gL1sh,ok_gL2sh),
            AND(ok_gL1pad,ok_gL2pad), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_copad_sh_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh,ok_gL2sh),
            AND(ok_gL2pad,ok_gL1pad), "P", kBlue)



  draw_geff(targetDir, "eff_lx_track_dg_gem_l1_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1or2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1and2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)

  draw_geff(targetDir, "eff_lx_track_dg_gem_l1_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1or2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1and2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)


  draw_geff(targetDir, "eff_ly_track_dg_gem_l1_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1or2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1and2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)

  draw_geff(targetDir, "eff_ly_track_dg_gem_l1_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1or2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), OR(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1and2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM Digi in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), AND(ok_trk_gL1dg,ok_trk_gL2dg), "P", kBlue)
