from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
def cscSimHitOccupancyXY(plotter,i):
  
  ## per station
  draw_occ(plotter.targetDir, "sh_csc_xy_st1" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st2" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st3" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,plotter.sel[i]), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st4" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station4;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,plotter.sel[i]), "COLZ")

  ## per station even/odd
  draw_occ(plotter.targetDir, "sh_csc_xy_st1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st2_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st3_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st4_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station4;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station1;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station2;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station3;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st3,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: station4;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(st4,plotter.sel[i],odd), "COLZ")

  ## per endcap, station, even/odd
  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st1,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st2,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st2_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st2,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st3,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st3_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st3,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station4, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st4,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st4_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station4, Odd;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st4,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st1,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st2_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st2,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st2_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st2,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st3_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st3,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st3_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st3,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st4_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station4, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec1,st4,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st4_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station4, Even;globalX [cm];globalY [cm]",
           "h_", "(350,-700,700,350,-700,700)", "globalX:globalY", AND(ec2,st4,plotter.sel[i],even), "COLZ")


  ## per endcap, station, ring1 even/odd
  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st1_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, ring1, Odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st1,ri1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st1_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, ring1, Odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st1,ri1,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st2_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, ring1, odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st2,ri1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st2_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, ring1, odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st2,ri1,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st3_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, ring1, odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st3,ri1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st3_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, ring1, odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st3,ri1,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st4_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station4, ring1, odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st4,ri1,plotter.sel[i],odd), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st4_ri1_odd" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station4, ring1, odd;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st4,ri1,plotter.sel[i],odd), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st1_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station1, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st1,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st1_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station1, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st1,ri1,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st2_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station2, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st2,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st2_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station2, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st2,ri1,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st3_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station3, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st3,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st3_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station3, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st3,ri1,plotter.sel[i],even), "COLZ")

  draw_occ(plotter.targetDir, "sh_csc_xy_ec1_st4_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region1, station4, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec1,st4,ri1,plotter.sel[i],even), "COLZ")
  draw_occ(plotter.targetDir, "sh_csc_xy_ec2_st4_ri1_even" + plotter.suff[i], plotter.ext, plotter.treeCSCSimHits,
           plotter.pre[i] + " SimHit occupancy: region-1, station4, ring1, even;globalX [cm];globalY [cm]",
           "h_", "(200,-400,400,200,-400,400)", "globalX:globalY", AND(ec2,st4,ri1,plotter.sel[i],even), "COLZ")


#_______________________________________________________________________________
def cscSimHitOccupancyRZ():
  pass

#_______________________________________________________________________________
def cscSimHitTOF():
  pass
