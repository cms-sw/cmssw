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
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st4" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station4;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st4,plotter.sel[i]), "COLZ")

  ## per endcap, station
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station1;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st1,rm1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station2;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st2,rm1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station3;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st3,rm1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1,station4;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st4,rm1,plotter.sel[i]), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station1;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st1,rp1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station2;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st2,rp1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station3;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st3,rp1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1,station4;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st4,rp1,plotter.sel[i]), "COLZ")

  """
  ## per station even/odd
  draw_occ(plotter.targetDir, "sh_rpc_xy_st1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station1;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st3_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st3,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st4_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station4;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st4,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station1;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st3,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: station4;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(st4,plotter.sel[i],odd), "COLZ")
  """

  """         
  ## per endcap, station, even/odd
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st1,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st2,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st3,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st3,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station4, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st4,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station4, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st4,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st1,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st2,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st3_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st3,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st3_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st3,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st4_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station4, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st4,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st4_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station4, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st4,plotter.sel[i],even), "COLZ")

  ## per endcap, station, l1 even/odd
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2 ,layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st2,plotter.sel[i],odd), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_l1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st2_l2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_l1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2 ,layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st2_l2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st2,plotter.sel[i],even), "COLZ")


  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_l1_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2 ,layer1, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_l2_odd" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer2, Odd;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st1,plotter.sel[i],odd), "COLZ")
  
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_l1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rm1_st1_l2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rm1,st1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_l1_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2 ,layer1, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_rpc_xy_rp1_st1_l2_even" + plotter.suff[i], plotter.ext, plotter.treeRPCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, layer2, Even;globalX [cm];globalY [cm]",
           "h_", "(100,-700,700,100,-700,700)", "globalX:globalY", AND(rp1,st1,plotter.sel[i],even), "COLZ")
  """
  
#_______________________________________________________________________________
def rpcSimHitOccupancyRZ():
  pass

#_______________________________________________________________________________
def rpcSimHitTOF():
  pass
