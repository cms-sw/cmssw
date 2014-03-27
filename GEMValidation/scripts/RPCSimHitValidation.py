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
def rpcSimHitOccupancyXY(plotter,i):
  
  ## per station
  draw_occ(plotter.targetDir, "sh_rpc_xy_st1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st4" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station4;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,plotter.sel[i]), "COLZ")

  ## per endcap, station
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rm1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station2;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rm1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rm1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rm1,plotter.sel[i]), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rp1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station2;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rp1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rp1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rp1,plotter.sel[i]), "COLZ")

  ## per endcap, station
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station1,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rp1,even,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station2,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rp1,even,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rp1,even,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rp1,even,plotter.sel[i]), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station1,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rm1,even,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station2,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rm1,even,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rm1,even,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4,even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rm1,even,plotter.sel[i]), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station1,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rp1,odd,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station2,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rp1,odd,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rp1,odd,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rp1,odd,plotter.sel[i]), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station1,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rm1,odd,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station2,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rm1,odd,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rm1,odd,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4,odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rm1,odd,plotter.sel[i]), "COLZ")

  ## per endcap, station, ring
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_ri1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,200,400,-400,400)", "globalX:globalY", AND(st3,rm1,ri1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_ri1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,200,400,-400,400)", "globalX:globalY", AND(st4,rm1,ri1,plotter.sel[i]), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station1,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rm1,ri2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station2,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rm1,ri2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,100,-700,700)", "globalX:globalY", AND(st3,rm1,ri2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st4,rm1,ri2,plotter.sel[i]), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station1,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rm1,ri3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station2,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rm1,ri3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rm1,ri3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rm1,ri3,plotter.sel[i]), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_ri1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st3,rp1,ri1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_ri1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st4,rp1,ri1,plotter.sel[i]), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station1,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rp1,ri2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station2,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rp1,ri2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rp1,ri2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_ri2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rp1,ri2,plotter.sel[i]), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station1,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,rp1,ri3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station2,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,rp1,ri3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,rp1,ri3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_ri3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,rp1,ri3,plotter.sel[i]), "COLZ")

  ## per endcap, station, ring, chamber
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st3,rp1,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st3,rp1,ri1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st3,rm1,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st3,rm1,ri1,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st4,rp1,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st4,rp1,ri1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st4,rm1,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4,ring1;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(st4,rm1,ri1,plotter.sel[i],odd), "COLZ")

  
#_______________________________________________________________________________
def rpcSimHitOccupancyRZ():
  pass

#_______________________________________________________________________________
def rpcSimHitTOF():
  pass
