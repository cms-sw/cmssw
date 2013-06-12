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

  inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/gem_rh_ana.test.root'
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
  simTracks = "Tracks"

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
    
  draw_occ(targetDir, "localrh_xy_rm1_l1", ext, treeHits, " SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rm1.GetTitle(), l1.GetTitle())), "COLZ");
  draw_occ(targetDir, "localrh_xy_rm1_l2", ext, treeHits, " SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rm1.GetTitle(), l2.GetTitle())), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_l1", ext, treeHits, " SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rp1.GetTitle(), l1.GetTitle())), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_l2", ext, treeHits, " SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s" %(rp1.GetTitle(), l2.GetTitle())), "COLZ");
  
  draw_occ(targetDir, "localrh_zr_rm1", ext, treeHits, " SimHit occupancy: region-1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,-573,-564,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rm1, "COLZ");
  draw_occ(targetDir, "localrh_zr_rp1", ext, treeHits, " SimHit occupancy: region1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,564,573,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rp1, "COLZ");
  
