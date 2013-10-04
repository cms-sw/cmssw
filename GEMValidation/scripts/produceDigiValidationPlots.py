import sys

from ROOT import TFile,TStyle,TKey,TTree,TH1F,TH2D
from ROOT import TMath,TCanvas,TCut
from ROOT import gStyle,gROOT,gPad
from ROOT import kBlue

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)



if __name__ == "__main__":  

  inputFile = str(sys.argv[1])
  if len(inputFile) < 3:
      inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/gem_digi_ana.root'
  targetDir = './'
  ## extension for figures - add more?
  ext = ".png"
  
  ## npads
  npads = 96

  ## Trees
  analyzer = "GEMDigiAnalyzer"
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
  draw_occ(targetDir, "strip_dg_xy_rm1_l1", ext, treeDigis, "Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rm1_l2", ext, treeDigis, "Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_l1", ext, treeDigis, "Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_dg_xy_rp1_l2", ext, treeDigis, "Digi occupancy: region1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "COLZ") 

  draw_occ(targetDir, "strip_dg_zr_rm1", ext, treeDigis, "Digi occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(targetDir, "strip_dg_zr_rp1", ext, treeDigis, "Digi occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")

  draw_occ(targetDir, "strip_dg_phistrip_rm1_l1", ext, treeDigis, "Digi occupancy: region-1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rm1_l2", ext, treeDigis, "Digi occupancy: region-1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_l1", ext, treeDigis, "Digi occupancy: region1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_dg_phistrip_rp1_l2", ext, treeDigis, "Digi occupancy: region1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "COLZ")
 
  draw_1D(targetDir, "strip_dg_rm1_l1", ext, treeDigis, "Digi occupancy per strip number, region-1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())))
  draw_1D(targetDir, "strip_dg_rm1_l2", ext, treeDigis, "Digi occupancy per strip number, region-1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())))
  draw_1D(targetDir, "strip_dg_rp1_l1", ext, treeDigis, "Digi occupancy per strip number, region1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())))
  draw_1D(targetDir, "strip_dg_rp1_l2", ext, treeDigis, "Digi occupancy per strip number, region1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())))
  
  ## Bunch crossing plots
  draw_bx(targetDir, "strip_digi_bx_rm1_l1", ext, treeDigis, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())))
  draw_bx(targetDir, "strip_digi_bx_rm1_l2", ext, treeDigis, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())))
  draw_bx(targetDir, "strip_digi_bx_rp1_l1", ext, treeDigis, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())))
  draw_bx(targetDir, "strip_digi_bx_rp1_l2", ext, treeDigis, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())))

  treePads = dirAna.Get(pads)
  if not treePads:
    sys.exit('Tree %s does not exist.' %(treePads))

  ## occupancy plots
  draw_occ(targetDir, "pad_dg_xy_rm1_l1", ext, treePads, "Pad occupancy: region-1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "pad_dg_xy_rm1_l2", ext, treePads, "Pad occupancy: region-1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "COLZ")
  draw_occ(targetDir, "pad_dg_xy_rp1_l1", ext, treePads, "Pad occupancy: region1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "pad_dg_xy_rp1_l2", ext, treePads, "Pad occupancy: region1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "COLZ")

  draw_occ(targetDir, "pad_dg_zr_rm1", ext, treePads, "Pad occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ")
  draw_occ(targetDir, "pad_dg_zr_rp1", ext, treePads, "Pad occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ")

  draw_occ(targetDir, "pad_dg_phipad_rm1_l1", ext, treePads, "Pad occupancy: region-1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npads/2.) + ",0, %f" %(npads) + ")", "pad:g_phi", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rm1_l2", ext, treePads, "Pad occupancy: region-1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npads/2.) + ",0, %f" %(npads) + ")", "pad:g_phi", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_l1", ext, treePads, "Pad occupancy: region1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npads/2.) + ",0, %f" %(npads) + ")", "pad:g_phi", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "pad_dg_phipad_rp1_l2", ext, treePads, "Pad occupancy: region1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npads/2.) + ",0, %f" %(npads) + ")", "pad:g_phi", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "COLZ")
 
  draw_1D(targetDir, "pad_dg_rm1_l1", ext, treePads, "Digi occupancy per pad number, region-1 layer1;pad number;entries", 
	  "h_", "( %f" %(npads) + ",0.5, %f" %(npads + 0.5) + ")", "pad", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())))
  draw_1D(targetDir, "pad_dg_rm1_l2", ext, treePads, "Digi occupancy per pad number, region-1 layer2;pad number;entries", 
	  "h_", "( %f" %(npads) + ",0.5, %f" %(npads + 0.5) + ")", "pad", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())))
  draw_1D(targetDir, "pad_dg_rp1_l1", ext, treePads, "Digi occupancy per pad number, region1 layer1;pad number;entries", 
	  "h_", "( %f" %(npads) + ",0.5, %f" %(npads + 0.5) + ")", "pad", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())))
  draw_1D(targetDir, "pad_dg_rp1_l2", ext, treePads, "Digi occupancy per pad number, region1 layer2;pad number;entries", 
	  "h_", "( %f" %(npads) + ",0.5, %f" %(npads + 0.5) + ")", "pad", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())))

  ## Bunch crossing plots
  draw_bx(targetDir, "pad_dg_bx_rm1_l1", ext, treePads, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())))
  draw_bx(targetDir, "pad_dg_bx_rm1_l2", ext, treePads, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())))
  draw_bx(targetDir, "pad_dg_bx_rp1_l1", ext, treePads, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())))
  draw_bx(targetDir, "pad_dg_bx_rp1_l2", ext, treePads, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())))

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
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npads/2.) + ",0, %f" %(npads) + ")", "pad:g_phi", rm1, "COLZ")
  draw_occ(targetDir, "copad_dg_phipad_rp1_l1", ext, treeCoPads, "Pad occupancy: region1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654, %f" %(npads/2.) + ",0, %f" %(npads) + ")", "pad:g_phi", rp1, "COLZ")
 
  draw_1D(targetDir, "copad_dg_rm1_l1", ext, treeCoPads, "Digi occupancy per pad number, region-1;pad number;entries", 
	  "h_", "( %f" %(npads) + ",0.5, %f" %(npads + 0.5) +  ")", "pad", rm1)
  draw_1D(targetDir, "copad_dg_rp1_l1", ext, treeCoPads, "Digi occupancy per pad number, region1;pad number;entries", 
	  "h_", "( %f" %(npads) + ",0.5, %f" %(npads + 0.5) +  ")", "pad", rp1)

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
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s || %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s && %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_dg_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, TCut("%s || %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, TCut("%s && %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)

  ## digis with matched simhits
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut("%s || %s" %(ok_gL2sh.GetTitle(), ok_gL1sh.GetTitle())),
            TCut("%s || %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_dg_sh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut("%s && %s" %(ok_gL2sh.GetTitle(), ok_gL1sh.GetTitle())),
            TCut("%s && %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_dg_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && %s" %(ok_eta.GetTitle(), ok_gL1sh.GetTitle())), ok_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && %s" %(ok_eta.GetTitle(), ok_gL2sh.GetTitle())), ok_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && (%s || %s)"%(ok_eta.GetTitle(),ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s || %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_dg_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && (%s && %s)"%(ok_eta.GetTitle(),ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s && %s" %(ok_gL2dg.GetTitle(), ok_gL1dg.GetTitle())), "P", kBlue)

  ## pads
  draw_geff(targetDir, "eff_eta_track_pad_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s || %s"%(ok_gL2pad.GetTitle(), ok_gL1pad.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_pad_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta,
            TCut("%s || %s"%(ok_gL2pad.GetTitle(), ok_gL1pad.GetTitle())), "P", kBlue)

  ## pads with matched simhits
  draw_geff(targetDir, "eff_eta_track_pad_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_pad_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut("%s || %s"%(ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s || %s" %(ok_gL2pad.GetTitle(),ok_gL1pad.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_pad_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && %s" %(ok_eta.GetTitle(),ok_gL1sh.GetTitle())), ok_gL1pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && %s" %(ok_eta.GetTitle(),ok_gL2sh.GetTitle())), ok_gL2pad, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_pad_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && (%s || %s)" %(ok_eta.GetTitle(),ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s || %s" %(ok_gL2pad.GetTitle(),ok_gL1pad.GetTitle())), "P", kBlue)

  ## copads
  draw_geff(targetDir, "eff_eta_track_copad_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s && %s" %(ok_gL1pad.GetTitle(),ok_gL2pad.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_copad_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta,TCut("%s && %s"%(ok_gL1pad.GetTitle(), ok_gL2pad.GetTitle())), "P", kBlue)

  ## copads with matched simhits
  draw_geff(targetDir, "eff_eta_track_copad_sh_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut("%s && %s" %(ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s && %s" %(ok_gL1pad.GetTitle(),ok_gL2pad.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_copad_sh_gem", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && %s && %s" %(ok_eta.GetTitle(),ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s && %s" %(ok_gL1pad.GetTitle(),ok_gL2pad.GetTitle())), "P", kBlue)



  draw_geff(targetDir, "eff_lx_track_dg_gem_l1_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1or2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), TCut("%s || %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1and2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), TCut("%s && %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_lx_track_dg_gem_l1_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1or2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), TCut("%s || %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_dg_gem_l1and2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), TCut("%s && %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)


  draw_geff(targetDir, "eff_ly_track_dg_gem_l1_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1or2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), TCut("%s || %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1and2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", TCut(""), TCut("%s && %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_ly_track_dg_gem_l1_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), ok_trk_gL1dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), ok_trk_gL2dg, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1or2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), TCut("%s || %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_dg_gem_l1and2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM DigiHit in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", TCut(""), TCut("%s && %s" %(ok_trk_gL1dg.GetTitle(),ok_trk_gL2dg.GetTitle())), "P", kBlue)
