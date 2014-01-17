import sys

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

if __name__ == "__main__":  

  inputFile = 'gem_localrec_ana.root'
  outputFile = 'gem_localrec_ana_tmp.root'
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

  fileOut = TFile.Open(outputFile, "RECREATE")

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_geff2(targetDir, "recHitEfficiencyGlobalPhi", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. phi;#phi", 
	   "h_", "(50,-TMath::Pi,+TMath::Pi)", "globalPhi", TCut("station==1"), TCut("station==1"), "P", kBlue);

  draw_geff2(targetDir, "recHitEfficiencyPerChamber_st1", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber (station 1);chamber", 
	   "h_", "(36,0.5,36.5)", "chamber", TCut("station==1"), TCut("station==1"), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_st2", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber (station 2);chamber", 
	   "h_", "(18,0.5,18.5)", "chamber", TCut("station==2"), TCut("station==2"), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_st3", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber (station 3);chamber", 
	   "h_", "(18,0.5,18.5)", "chamber", TCut("station==3"), TCut("station==3"), "P", kBlue);

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_1D(targetDir, "clsDistribution", ext, treeHits, "CLS; CLS; entries", 
	   "h_", "(11,-0.5,10.5)", "clusterSize", TCut(""), "");

  draw_1D(targetDir, "clsDistribution_rm1_l1", ext, treeHits, "CLS region-1, layer1; CLS; entries", 
	   "h_", "(11,-0.5,10.5)", "clusterSize", AND(rm1,l1), "");
  draw_1D(targetDir, "clsDistribution_rm1_l2", ext, treeHits, "CLS region-1, layer2; CLS; entries", 
	   "h_", "(11,-0.5,10.5)", "clusterSize", AND(rp1,l2), "");
  draw_1D(targetDir, "clsDistribution_rp1_l1", ext, treeHits, "CLS region1, layer1; CLS; entries", 
	   "h_", "(11,-0.5,10.5)", "clusterSize", AND(rm1,l1), "");
  draw_1D(targetDir, "clsDistribution_rp1_l2", ext, treeHits, "CLS region1, layer2; CLS; entries", 
	   "h_", "(11,-0.5,10.5)", "clusterSize", AND(rp1,l2), "");

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_1D(targetDir, "bxDistribution", ext, treeHits, "BX; BX; entries", 
	   "h_", "(11,-5.5,5.5)", "bx", TCut(""), "");

  draw_1D(targetDir, "bxDistribution_st1", ext, treeHits, "BX (station 1); BX; entries", 
	   "h_", "(11,-5.5,5.5)", "bx", TCut("station==1"), "");
  draw_1D(targetDir, "bxDistribution_st2", ext, treeHits, "BX (station 2); BX; entries", 
	   "h_", "(11,-5.5,5.5)", "bx", TCut("station==2"), "");
  draw_1D(targetDir, "bxDistribution_st3", ext, treeHits, "BX (station 3); BX; entries", 
	   "h_", "(11,-5.5,5.5)", "bx", TCut("station==3"), "");

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_1D(targetDir, "recHitPullX", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec}; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", TCut(""), "");

  draw_1D(targetDir, "recHitPullX_rm1_l1", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region-1, layer1; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", AND(rm1,l1), "");
  draw_1D(targetDir, "recHitPullX_rm1_l2", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region-1, layer2; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", AND(rm1,l2), "");
  draw_1D(targetDir, "recHitPullX_rp1_l1", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region+1, layer1; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", AND(rp1,l1), "");
  draw_1D(targetDir, "recHitPullX_rp1_l2", ext, treeHits, "(x_{sim} - x_{rec}) / #sigma_{rec} region'1, layer2; (x_{sim} - x_{rec}) / #sigma_{rec}; entries", 
	   "h_", "(100,-50,+50)", "pull", AND(rp1,l2), "");

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_1D(targetDir, "recHitDPhi", ext, treeHits, "#phi_{rec} - #phi_{sim}; #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut(""), "");

  draw_1D(targetDir, "recHitDPhi_st1", ext, treeHits, "#phi_{rec} - #phi_{sim}; #phi_{rec} - #phi_{sim} [rad] (station 1); entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==1"), "");
  draw_1D(targetDir, "recHitDPhi_st2", ext, treeHits, "#phi_{rec} - #phi_{sim}; #phi_{rec} - #phi_{sim} [rad] (station 2); entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==2"), "");
  draw_1D(targetDir, "recHitDPhi_st3", ext, treeHits, "#phi_{rec} - #phi_{sim}; #phi_{rec} - #phi_{sim} [rad] (station 3); entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==3"), "");

  draw_1D(targetDir, "recHitDPhi_rm1_l1", ext, treeHits, "#phi_{rec} - #phi_{sim} region-1, layer1; #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", AND(rm1,l1), "");
  draw_1D(targetDir, "recHitDPhi_rm1_l2", ext, treeHits, "#phi_{rec} - #phi_{sim} region-1, layer2; #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", AND(rm1,l2), "");
  draw_1D(targetDir, "recHitDPhi_rp1_l1", ext, treeHits, "#phi_{rec} - #phi_{sim} region1, layer1; #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", AND(rp1,l1), "");
  draw_1D(targetDir, "recHitDPhi_rp1_l2", ext, treeHits, "#phi_{rec} - #phi_{sim} region1, layer2; #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", AND(rp1,l2), "");

  draw_1D(targetDir, "recHitDPhi_st1_cls1", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 1, cls = 1); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==1 && clusterSize==1"), "");
  draw_1D(targetDir, "recHitDPhi_st1_cls2", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 1, cls = 2); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==1 && clusterSize==2"), "");
  draw_1D(targetDir, "recHitDPhi_st1_cls3", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 1, cls = 3); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==1 && clusterSize==3"), "");

  draw_1D(targetDir, "recHitDPhi_st2_cls1", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 2, cls = 1); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==2 && clusterSize==1"), "");
  draw_1D(targetDir, "recHitDPhi_st2_cls2", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 2, cls = 2); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==2 && clusterSize==2"), "");
  draw_1D(targetDir, "recHitDPhi_st2_cls3", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 2, cls = 3); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==2 && clusterSize==3"), "");

  draw_1D(targetDir, "recHitDPhi_st3_cls1", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 3, cls = 1); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==3 && clusterSize==1"), "");
  draw_1D(targetDir, "recHitDPhi_st3_cls2", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 3, cls = 2); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==3 && clusterSize==2"), "");
  draw_1D(targetDir, "recHitDPhi_st3_cls3", ext, treeHits, "#phi_{rec} - #phi_{sim} (station 3, cls = 3); #phi_{rec} - #phi_{sim} [rad]; entries", 
	   "h_", "(100,-0.001,+0.001)", "(globalPhi - globalPhi_sim)", TCut("station==3 && clusterSize==3"), "");

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rm1_l1", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region-1, layer1;chamber", 
	   "h_", "(38,0,38)", "chamber", AND(rm1,l1,st1),AND(rm1,l1,st1), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rm1_l2", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region-1, layer2;chamber", 
	   "h_", "(38,0,38)", "chamber", AND(rm1,l2,st1),AND(rm1,l2,st1), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rp1_l1", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region+1, layer1;chamber", 
	   "h_", "(38,0,38)", "chamber", AND(rp1,l1,st1),AND(rp1,l1,st1), "P", kBlue);
  draw_geff2(targetDir, "recHitEfficiencyPerChamber_rp1_l2", ext, treeHits, treeSimHits, "Local Reco Efficiency vs. chamber : region+1, layer2;chamber", 
	   "h_", "(38,0,38)", "chamber", AND(rp1,l2,st1),AND(rp1,l2,st1), "P", kBlue);

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_occ(targetDir, "localrh_xy_rm1_st1_l1", ext, treeHits, " GEM RecHit occupancy: region-1, station1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(200,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,l1,st1), "COLZ");
  draw_occ(targetDir, "localrh_xy_rm1_st1_l2", ext, treeHits, " GEM RecHit occupancy: region-1, station1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(200,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,l2,st1), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_st1_l1", ext, treeHits, " GEM RecHit occupancy: region+1, station1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(200,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,l1,st1), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_st1_l2", ext, treeHits, " GEM RecHit occupancy: region+1, station1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(200,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,l2,st1), "COLZ");

  draw_occ(targetDir, "localrh_xy_rm1_st2_l1", ext, treeHits, " GEM RecHit occupancy: region-1, station2, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rm1,l1,st2), "COLZ");
  draw_occ(targetDir, "localrh_xy_rm1_st2_l2", ext, treeHits, " GEM RecHit occupancy: region-1, station2, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rm1,l2,st2), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_st2_l1", ext, treeHits, " GEM RecHit occupancy: region+1, station2, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rp1,l1,st2), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_st2_l2", ext, treeHits, " GEM RecHit occupancy: region+1, station2, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rm1,l2,st2), "COLZ");

  draw_occ(targetDir, "localrh_xy_rm1_st3_l1", ext, treeHits, " GEM RecHit occupancy: region-1, station3, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rm1,l1,st3), "COLZ");
  draw_occ(targetDir, "localrh_xy_rm1_st3_l2", ext, treeHits, " GEM RecHit occupancy: region-1, station3, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rm1,l2,st3), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_st3_l1", ext, treeHits, " GEM RecHit occupancy: region+1, station3, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rp1,l1,st3), "COLZ");
  draw_occ(targetDir, "localrh_xy_rp1_st3_l2", ext, treeHits, " GEM RecHit occupancy: region+1, station3, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(200,-360,360,200,-360,360)", "globalY:globalX", AND(rm1,l2,st3), "COLZ");
  
  draw_occ(targetDir, "localrh_zr_rm1_st1", ext, treeHits, " GEM RecHit occupancy: region-1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,-573,-564,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rm1, "COLZ");
  draw_occ(targetDir, "localrh_zr_rp1_st1", ext, treeHits, " GEM RecHit occupancy: region1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,564,573,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rp1, "COLZ");

  draw_occ(targetDir, "localrh_zr_rm1_st23", ext, treeHits, " GEM RecHit occupancy: region-1;globalZ [cm];globalR [cm]", 
	   "h_", "(300,-805,-785,220,130,350)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rm1, "COLZ");
  draw_occ(targetDir, "localrh_zr_rp1_st23", ext, treeHits, " GEM RecHit occupancy: region1;globalZ [cm];globalR [cm]", 
	   "h_", "(300,785,805,220,130,350)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rp1, "COLZ");

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_occ(targetDir, "strip_rh_phistrip_rm1_st1_l1", ext, treeHits, "GEM RecHit occupancy: region-1 layer1 station1; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", AND(rm1,l1,st1), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rm1_st1_l2", ext, treeHits, "GEM RecHit occupancy: region-1 layer2 station1; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", AND(rm1,l2,st1), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_st1_l1", ext, treeHits, "GEM RecHit occupancy: region1 layer1 station1; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", AND(rp1,l1,st1), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_st1_l2", ext, treeHits, "GEM RecHit occupancy: region1 layer2 station1; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "firstClusterStrip:globalPhi", AND(rp1,l2,st1), "COLZ")

  draw_occ(targetDir, "strip_rh_phistrip_rm1_st2_l1", ext, treeHits, "GEM RecHit occupancy: region-1 layer1 station2; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rm1,l1,st2), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rm1_st2_l2", ext, treeHits, "GEM RecHit occupancy: region-1 layer2 station2; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rm1,l2,st2), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_st2_l1", ext, treeHits, "GEM RecHit occupancy: region1 layer1 station2; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rp1,l1,st2), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_st2_l2", ext, treeHits, "GEM RecHit occupancy: region1 layer2 station2; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rp1,l2,st2), "COLZ")

  draw_occ(targetDir, "strip_rh_phistrip_rm1_st3_l1", ext, treeHits, "GEM RecHit occupancy: region-1 layer1 station3; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rm1,l1,st3), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rm1_st3_l2", ext, treeHits, "GEM RecHit occupancy: region-1 layer2 station3; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rm1,l2,st3), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_st3_l1", ext, treeHits, "GEM RecHit occupancy: region1 layer1 station3; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rp1,l1,st3), "COLZ")
  draw_occ(targetDir, "strip_rh_phistrip_rp1_st3_l2", ext, treeHits, "GEM RecHit occupancy: region1 layer2 station3; #phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,384,0,768)", "firstClusterStrip:globalPhi", AND(rp1,l2,st3), "COLZ")

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_1D_adv(targetDir, "strip_rh_tot", ext, treeHits, "GEM RecHit occupancy per strip number;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", TCut(""))

  draw_1D_adv(targetDir, "strip_rh_rm1_st1_l1_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer1 station1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rm1,l1,st1))
  draw_1D_adv(targetDir, "strip_rh_rm1_st1_l2_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer2 station1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rm1,l2,st1))
  draw_1D_adv(targetDir, "strip_rh_rp1_st1_l1_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer1 station1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rp1,l1,st1))
  draw_1D_adv(targetDir, "strip_rh_rp1_st1_l2_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer2 station1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rp1,l2,st1))

  draw_1D_adv(targetDir, "strip_rh_rm1_st2_l1_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer1 station2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rm1,l1,st2))
  draw_1D_adv(targetDir, "strip_rh_rm1_st2_l2_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer2 station2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rm1,l2,st2))
  draw_1D_adv(targetDir, "strip_rh_rp1_st2_l1_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer1 station2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rp1,l1,st2))
  draw_1D_adv(targetDir, "strip_rh_rp1_st2_l2_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer2 station2;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rp1,l2,st2))

  draw_1D_adv(targetDir, "strip_rh_rm1_st3_l1_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer1 station3;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rm1,l1,st3))
  draw_1D_adv(targetDir, "strip_rh_rm1_st3_l2_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer2 station3;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rm1,l2,st3))
  draw_1D_adv(targetDir, "strip_rh_rp1_st3_l1_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer1 station3;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rp1,l1,st3))
  draw_1D_adv(targetDir, "strip_rh_rp1_st3_l2_tot", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer2 station3;strip number;entries", 
	  "h_", "(768,0.5,768.5)", "firstClusterStrip", AND(rp1,l2,st3))

#-------------------------------------------------------------------------------------------------------------------------------------#

  draw_2D_adv(targetDir, "roll_vs_strip_rh", ext, treeHits, "GEM RecHit occupancy per roll and strip number;strip number;roll", 
	  "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", TCut(""), "COLZ")

  draw_2D_adv(targetDir, "roll_vs_strip_rh_rm1_st1_l1", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region-1 layer1 station1;strip number;roll", "h_", "(384,0.5,384.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rm1,l1,st1), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rm1_st1_l2", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region-1 layer2 station1;strip number;roll", "h_", "(384,0.5,384.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rm1,l2,st1), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rp1_st1_l1", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region1 layer1 station1;strip number;roll", "h_", "(384,0.5,384.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rp1,l1,st1), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rp1_st1_l2", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region1 layer2 station1;strip number;roll", "h_", "(384,0.5,384.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rp1,l2,st1), "COLZ")

  draw_2D_adv(targetDir, "roll_vs_strip_rh_rm1_st2_l1", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region-1 layer1 station2;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rm1,l1,st2), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rm1_st2_l2", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region-1 layer2 station2;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rm1,l2,st2), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rp1_st2_l1", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region1 layer1 station2;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rp1,l1,st2), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rp1_st2_l2", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region1 layer2 station2;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rp1,l2,st2), "COLZ")

  draw_2D_adv(targetDir, "roll_vs_strip_rh_rm1_st3_l1", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region-1 layer1 station3;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rm1,l1,st3), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rm1_st3_l2", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region-1 layer2 station3;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rm1,l2,st3), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rp1_st3_l1", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region1 layer1 station3;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rp1,l1,st3), "COLZ")
  draw_2D_adv(targetDir, "roll_vs_strip_rh_rp1_st3_l2", ext, treeHits, "GEM RecHit occupancy per roll and strip number, region1 layer2 station3;strip number;roll", "h_", "(768,0.5,768.5,12,0.5,12.5)", "firstClusterStrip:roll", AND(rp1,l2,st3), "COLZ")

  draw_1D(targetDir, "roll_rh_rm1_l1_st3", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer1 station3;strip number;entries", 
	  "h_", "(12,0.5,12.5)", "roll", AND(rm1,l1,st3))

  draw_1D(targetDir, "localX_rh_rm1_l1_st1", ext, treeHits, "GEM RecHit local x, region-1 layer1 station1;x [cm];entries", 
	  "h_", "(120,-60,60)", "x", AND(rm1,l1,st1))
  draw_1D(targetDir, "localX_rh_rm1_l1_st2", ext, treeHits, "GEM RecHit local x, region-1 layer1 station2;x [cm];entries", 
	  "h_", "(120,-60,60)", "x", AND(rm1,l1,st2))
  draw_1D(targetDir, "localX_rh_rm1_l1_st3", ext, treeHits, "GEM RecHit local x, region-1 layer1 station3;x [cm];entries", 
	  "h_", "(120,-60,60)", "x", AND(rm1,l1,st3))

#-------------------------------------------------------------------------------------------------------------------------------------#
 
  draw_1D(targetDir, "strip_rh_rm1_l1", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rm1,l1))
  draw_1D(targetDir, "strip_rh_rm1_l2", ext, treeHits, "GEM RecHit occupancy per strip number, region-1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rm1,l2))
  draw_1D(targetDir, "strip_rh_rp1_l1", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rp1,l1))
  draw_1D(targetDir, "strip_rh_rp1_l2", ext, treeHits, "GEM RecHit occupancy per strip number, region1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "firstClusterStrip", AND(rp1,l2))

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
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), OR(ok_gL2rh,ok_gL1rh), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), AND(ok_gL2rh,ok_gL1rh), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_rh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, OR(ok_gL2rh,ok_gL1rh), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, AND(ok_gL2rh,ok_gL1rh), "P", kBlue)

  ## recHits with matched simhits
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1rh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2rh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", OR(ok_gL2sh,ok_gL1sh),
            OR(ok_gL2rh,ok_gL1rh), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_rh_sh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", AND(ok_gL2sh,ok_gL1sh),
            AND(ok_gL2rh,ok_gL1rh), "P", kBlue)

  draw_geff(targetDir, "eff_phi_track_rh_gem_l1", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL1sh), ok_gL1rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,ok_gL2sh), ok_gL2rh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1or2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,OR(ok_gL1sh,ok_gL2sh)),
            OR(ok_gL2rh,ok_gL1rh), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_rh_gem_l1and2", ext, treeTracks, 
	    "Eff. for a SimTrack to have an associated GEM RecHit in l1 and l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", AND(ok_eta,AND(ok_gL1sh,ok_gL2sh)),
            AND(ok_gL2rh,ok_gL1rh), "P", kBlue)

  file.Close()
  fileOut.Close()
  
