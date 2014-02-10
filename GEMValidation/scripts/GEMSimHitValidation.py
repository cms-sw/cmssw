from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
def gemSimHitOccupancyXY(plotter,i):
  
  ## per station
  draw_occ(plotter.targetDir, "sh_gem_xy_st1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: station1;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(st1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_st2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(st2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_st3" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(st3,plotter.sel[i]), "COLZ")

  ## per station and per layer
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st1_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, layer1;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st1_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, layer2;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st1_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1 ,layer1;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st1_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, layer2;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l2,plotter.sel[i]), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st2_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer1;globalX [cm];globalY [cm]",
           "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st2,l1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st2_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer2;globalX [cm];globalY [cm]",
           "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st2,l2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st2_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer1;globalX [cm];globalY [cm]",
           "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st2,l1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st2_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer2;globalX [cm];globalY [cm]",
           "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st2,l2,plotter.sel[i]), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st3_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, layer1;globalX [cm];globalY [cm]",
           "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st3,l1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st3_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, layer2;globalX [cm];globalY [cm]",
           "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st3,l2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st3_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, layer1;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st3,l1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st3_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, layer2;globalX [cm];globalY [cm]",
           "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st3,l2,plotter.sel[i]), "COLZ")

  ## per station and per layer, odd/even
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st1_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st1_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st1_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1 ,layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st1_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l2,plotter.sel[i],odd), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st1_l1_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st1_l2_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st1_l1_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1 ,layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st1_l2_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l2,plotter.sel[i],even), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st2_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st2,l1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st2_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st2,l2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st2_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2 ,layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st2,l1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st2_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st2,l2,plotter.sel[i],odd), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st2_l1_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st2,l1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st2_l2_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st2,l2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st2_l1_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2 ,layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st2,l1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st2_l2_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st2,l2,plotter.sel[i],even), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st3_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st3,l1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st3_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st3,l2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st3_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3 ,layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st3,l1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st3_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st3,l2,plotter.sel[i],odd), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st3_l1_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st3,l1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rm1_st3_l2_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st3,l2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st3_l1_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3 ,layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st3,l1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_xy_rp1_st3_l2_even" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st3,l2,plotter.sel[i],even), "COLZ")

#_______________________________________________________________________________
def gemSimHitOccupancyRZ(plotter,i):
  draw_occ(plotter.targetDir, "sh_gem_zr_rm1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits, plotter.pre[i] + " SimHit occupancy: region-1;globalZ [cm];globalR [cm]",
           "h_", "(200,-573,-564,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rm1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_gem_zr_rp1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits, plotter.pre[i] + " SimHit occupancy: region1;globalZ [cm];globalR [cm]",
           "h_", "(200,564,573,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rp1,plotter.sel[i]), "COLZ")

#_______________________________________________________________________________
def gemSimHitTOF(plotter,i):
  draw_1D(plotter.targetDir, "sh_gem_tof_rm1_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits, plotter.pre[i] + " SimHit TOF: region-1, layer1;Time of flight [ns];entries", 
          "h_", "(40,18,22)", "timeOfFlight", AND(rm1,l1,plotter.sel[i]))
  draw_1D(plotter.targetDir, "sh_gem_tof_rm1_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits, plotter.pre[i] + " SimHit TOF: region-1, layer2;Time of flight [ns];entries", 
          "h_", "(40,18,22)", "timeOfFlight", AND(rm1,l2,plotter.sel[i]))
  draw_1D(plotter.targetDir, "sh_gem_tof_rp1_l1" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits, plotter.pre[i] + " SimHit TOF: region1, layer1;Time of flight [ns];entries", 
          "h_", "(40,18,22)", "timeOfFlight", AND(rp1,l1,plotter.sel[i]))
  draw_1D(plotter.targetDir, "sh_gem_tof_rp1_l2" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits, plotter.pre[i] + " SimHit TOF: region1, layer2;Time of flight [ns];entries", 
          "h_", "(40,18,22)", "timeOfFlight", AND(rp1,l2,plotter.sel[i]))

#_______________________________________________________________________________
def gemSimTrackToSimHitMatchingLX(plotter): 
  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l1_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, ok_trk_gL1sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, ok_trk_gL2sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l1or2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l1and2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", nocut, AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l1_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, ok_trk_gL1sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, ok_trk_gL2sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l1or2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_lx_track_sh_gem_l1and2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", nocut, AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)

#_______________________________________________________________________________
def gemSimTrackToSimHitMatchingLY(plotter): 
  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l1_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, ok_trk_gL1sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, ok_trk_gL2sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l1or2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l1and2_even", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)

  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l1_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, ok_trk_gL1sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, ok_trk_gL2sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l1or2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_ly_track_sh_gem_l1and2_odd", plotter.ext, plotter.treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)

#_______________________________________________________________________________
def gemSimTrackToSimHitMatchingEta(plotter): 
  draw_geff(plotter.targetDir, "eff_eta_track_sh_gem_l1or2", plotter.ext, plotter.treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, OR(ok_gL1sh,ok_gL2sh), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_sh_gem_l1", plotter.ext, plotter.treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, ok_gL1sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_sh_gem_l2", plotter.ext, plotter.treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, ok_gL2sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_eta_track_sh_gem_l1and2", plotter.ext, plotter.treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", nocut, AND(ok_gL1sh,ok_gL2sh), "P", kBlue)

#_______________________________________________________________________________
def gemSimTrackToSimHitMatchingPhi(plotter):  
  draw_geff(plotter.targetDir, "eff_phi_track_sh_gem_l1or2", plotter.ext, plotter.treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, OR(ok_gL1sh,ok_gL2sh), "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_sh_gem_l1", plotter.ext, plotter.treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, ok_gL1sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_sh_gem_l2", plotter.ext, plotter.treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, ok_gL2sh, "P", kBlue)
  draw_geff(plotter.targetDir, "eff_phi_track_sh_gem_l1and2", plotter.ext, plotter.treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, AND(ok_gL1sh,ok_gL2sh), "P", kBlue)

#_______________________________________________________________________________ 
def simhitMomentum(plotter,i): 
  c = TCanvas("c","c",600,600)
  c.Clear()
  plotter.treeGEMSimHits.Draw("pabs>>hh(200,0.,200.)",plotter.sel[i])
  h = TH1F(gDirectory.Get("hh"))
  gPad.SetLogx(0)
  gPad.SetLogy(1)
  h.SetTitle(plotter.pre[i] + " SimHits absolute momentum;Momentum [GeV/c];entries")       
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw("")        
  c.SaveAs(plotter.targetDir +"sh_momentum" + plotter.suff[i] + plotter.ext)

#_______________________________________________________________________________ 
def simhitMomentum(plotter,i): 
  draw_1D(plotter.targetDir, "sh_pdgid" + plotter.suff[i], plotter.ext, plotter.treeGEMSimHits,
          plotter.pre[i] + " SimHit PDG Id;PDG Id;entries", 
          "h_", "(200,-100.,100.)", "particleType", plotter.sel[i])

#_______________________________________________________________________________ 
def energyLoss(plotter,i):
  ## energy loss plot
  h = TH1F("h","",60,0.,6000.)
  entries = plotter.treeGEMSimHits.GetEntriesFast()
  for jentry in xrange(entries):
    ientry = plotter.treeGEMSimHits.LoadTree( jentry )
    if ientry < 0:
      break
    nb = plotter.treeGEMSimHits.GetEntry( jentry )
    if nb <= 0:
      continue
    if i==0:
      if abs(plotter.treeGEMSimHits.particleType)==13:
        h.Fill( plotter.treeGEMSimHits.energyLoss*1.e9 )
    elif i==1:
      if not abs(plotter.treeGEMSimHits.particleType)!=13:
        h.Fill( plotter.treeGEMSimHits.energyLoss*1.e9 )
    elif i==2:
      h.Fill( plotter.treeGEMSimHits.energyLoss*1.e9 )
        
  c = TCanvas("c","c",600,600)
  c.Clear()  
  h.SetTitle(plotter.pre[i] + " SimHit energy loss;Energy loss [eV];entries")
  gPad.SetLogx(0)
  gPad.SetLogy(0)
  h.SetMinimum(0.)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw("")        
  c.SaveAs(plotter.targetDir + "sh_energyloss" + plotter.suff[i] + plotter.ext)

#_______________________________________________________________________________ 
def etaOccupancy():
  ### FIX THIS ### 
  """
  GEM system settings
  nregion = 2
  nlayer = 2
  npart = 8
  
  eta partition; entries",4*npart,1.,1.+4*npart)
  eta occupancy plot
  h = TH1F("h", pre + " SimHit occupancy in eta partitions; occupancy in 
  entries = treeHits.GetEntriesFast()
  for jentry in xrange(entries):
  ientry = treeHits.LoadTree( jentry )
  if ientry < 0:
  break
  nb = treeHits.GetEntry( jentry )
  if nb <= 0:
  continue
  if treeHits.layer==2:
  layer = npart
  else:
  layer = 0
  if treeHits.region==1:
  region = 2.*npart
  else:
  region = 0
  if i==0:
  if abs(treeHits.particleType)==13:
  h.Fill(treeHits.roll + layer + region)
  elif i==1:
  if not abs(treeHits.particleType)!=13:
  h.Fill(treeHits.roll + layer + region)
  elif i==2:
  h.Fill(treeHits.roll + layer + region)
  
  c = TCanvas("c","c",600,600)
  c.Clear()  
  gPad.SetLogx(0)
  gPad.SetLogy(0)
  ibin = 1
  for iregion in range(1,3):
  if iregion ==1:
  region = "-"
  else:
  region = "+"
  for ilayer in range(1,3):
  for ipart in range(1,npart+1):
  h.GetXaxis().SetBinLabel(ibin,"%s%d%d"% (region,ilayer,ipart))
  ibin = ibin + 1
  h.SetMinimum(0.)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw("")        
  c.SaveAs(targetDir +"sh_globalEta" + suff + ext)
  """  
    
