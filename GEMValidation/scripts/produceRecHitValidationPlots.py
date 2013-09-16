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

  inputFile = '/cmshome/calabria/ValidationPlots2/CMSSW_6_1_2_SLHC6_patch1/src/GEMCode/GEMValidation/test/gem_localrec_ana.root'
  targetDir = './'
  
  ## extension for figures - add more?
  ext = ".png"
  
  ## GEM system settings
  nregion = 2
  nlayer = 2
  npart = 8
  
  ## Trees
  analyzer = "GEMRecHitAnalyzer"
  recHits = "GEMRecHitTree"
  simHits = "GEMSimHits"
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
    
  treeHits = dirAna.Get(recHits)
  if not treeHits:
    sys.exit('Tree %s does not exist.' %(treeHits))

  treeSimHits = dirAna.Get(simHits)
  if not treeSimHits:
    sys.exit('Tree %s does not exist.' %(treeSimHits))

  draw_geff2(targetDir, "recHitEfficiencyPerChamber", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber;chamber", 
	   "h_", "(38,0,38)", "chamber", TCut(""), TCut(""), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyGlobalPhi", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. phi;#phi", 
	   "h_", "(100,-TMath::Pi,+TMath::Pi)", "globalPhi", TCut(""), TCut(""), "P", kBlue);
  draw_1D(targetDir, "recHitPullX", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec}; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", TCut(""), "");

  draw_1D(targetDir, "recHitPullX_rm1_l1", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region-1, layer1; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "");
  draw_1D(targetDir, "recHitPullX_rm1_l2", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region-1, layer2; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "");
  draw_1D(targetDir, "recHitPullX_rp1_l1", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region+1, layer1; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "");
  draw_1D(targetDir, "recHitPullX_rp1_l2", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region'1, layer2; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "");

  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rm1_l1", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region-1, layer1;chamber", 
	   "h_", "(38,0,38)", "chamber", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())),TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rm1_l2", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region-1, layer2;chamber", 
	   "h_", "(38,0,38)", "chamber", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())),TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rp1_l1", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region+1, layer1;chamber", 
	   "h_", "(38,0,38)", "chamber", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())),TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rp1_l2", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region+1, layer2;chamber", 
	   "h_", "(38,0,38)", "chamber", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())),TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "P", kBlue);

  draw_occ(targetDir, "localrh_xy_rm1_l1", ext, treeHits, " GEM RecHit occupancy: region-1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "COLZ");
  draw_occ(targetDir, "localrh_xy_rm1_l2", ext, treeHits, " GEM RecHit occupancy: region-1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_l1", ext, treeHits, " GEM RecHit occupancy: region1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_l2", ext, treeHits, " GEM RecHit occupancy: region1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "COLZ");
  
  draw_occ(targetDir, "localrh_zr_rm1", ext, treeHits, " GEM RecHit occupancy: region-1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,-573,-564,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rm1, "COLZ");
  draw_occ(targetDir, "localrh_zr_rp1", ext, treeHits, " GEM RecHit occupancy: region1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,564,573,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rp1, "COLZ");

  draw_occ(targetDir, "strip_rh_phistrip_rm1_l1", ext, treeHits, "GEM RecHit occupancy: region-1 layer1; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rm1_l2", ext, treeHits, "GEM RecHit occupancy: region-1 layer2; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_l1", ext, treeHits, "GEM RecHit occupancy: region1 layer1; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_l2", ext, treeHits, "GEM RecHit occupancy: region1 layer2; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "COLZ")
 
  draw_1D(targetDir, "strip_rh_rm1_l1", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())))
  draw_1D(targetDir, "strip_rh_rm1_l2", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())))
  draw_1D(targetDir, "strip_rh_rp1_l1", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())))
  draw_1D(targetDir, "strip_rh_rp1_l2", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())))

  ## Tracks
  treeTracks = dirAna.Get(simTracks)
  if not treeTracks:
    sys.exit('Tree %s does not exist.' %(treeTracks))

  ## recHits
  draw_geff(targetDir, "eff_eta_track_rh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL1rh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL2rh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s || %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s && %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_rh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, TCut("%s || %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, TCut("%s && %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)

  ## recHits with matched simhits
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1rh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2rh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut("%s || %s" %(ok_gL2sh.GetTitle(), ok_gL1sh.GetTitle())),
            TCut("%s || %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut("%s && %s" %(ok_gL2sh.GetTitle(), ok_gL1sh.GetTitle())),
            TCut("%s && %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_rh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && %s" %(ok_eta.GetTitle(), ok_gL1sh.GetTitle())), ok_gL1rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && %s" %(ok_eta.GetTitle(), ok_gL2sh.GetTitle())), ok_gL2rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && (%s || %s)"%(ok_eta.GetTitle(),ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s || %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", TCut("%s && (%s && %s)"%(ok_eta.GetTitle(),ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())),
            TCut("%s && %s" %(ok_gL2rh.GetTitle(), ok_gL1rh.GetTitle())), "P", kBlue)
  
