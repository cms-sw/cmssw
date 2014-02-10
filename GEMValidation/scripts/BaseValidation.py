import sys

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

class SimHitPlotter():
  def __init__(self):
    self.inputDir = "/afs/cern.ch/user/d/dildick/work/GEM/testForGeometry/CMSSW_6_2_0_SLHC7/src/"
    self.inputFile = "gem_sh_ana.root"
    self.targetDir = "testDirectory/"
    self.ext = ".png"
    self.analyzer = "MuonSimHitAnalyzer"
    self.gemSimHits = "GEMSimHits"
    self.rpcSimHits = "RPCSimHits"
    self.cscSimHits = "CSCSimHits"
    self.me0SimHits = "ME0SimHits"
    self.simTracks = "Tracks"
    self.file = TFile.Open(self.inputDir + self.inputFile)
    self.dirAna = (self.file).Get(self.analyzer)
    self.treeGEMSimHits = (self.dirAna).Get(self.gemSimHits)
    self.treeRPCSimHits = (self.dirAna).Get(self.rpcSimHits)
    self.treeCSCSimHits = (self.dirAna).Get(self.cscSimHits)
    self.treeME0SimHits = (self.dirAna).Get(self.me0SimHits)
    self.treeTracks = (self.dirAna).Get(self.simTracks)
    self.sel = [muOnly,noMu,all]
    self.pre = ["Muon","Non muon","All"]
    self.suff = ["_muon","_nonmuon","_all"]
    self.geometry = "custom_GE11_6partitions_v1"

    
class DigiPlotter():
  def __init__(self):
    self.inputDir = "/afs/cern.ch/user/d/dildick/work/GEM/testForGeometry/CMSSW_6_2_0_SLHC7/src/"
    self.inputFile = "gem_digi_ana.root"
    self.targetDir = "testDirectory/"
    self.ext = ".png"
    self.analyzer = "MuonDigiAnalyzer"
    self.gemDigis = "GEMDigiTree"
    self.rpcDigis = "RPCDigiTree"
    self.gemCscPadDigis = "GEMCSCPadDigiTree"
    self.gemCscCoPadDigis = "GEMCSCCoPadDigiTree"
    self.simTracks = "TrackTree"
    self.file = TFile.Open(self.inputDir + self.inputFile)
    self.dirAna = (self.file).Get(self.analyzer)
    self.treeGEMDigis = (self.dirAna).Get(self.gemDigis)
    self.treeRPCDigis = (self.dirAna).Get(self.rpcDigis)
    self.treeGEMCSPadDigis = (self.dirAna).Get(self.gemCscPadDigis)
    self.treeGEMCSCoPadDigis = (self.dirAna).Get(self.gemCscCoPadDigis)
    self.treeTracks = (self.dirAna).Get(self.simTracks)
    self.nstripsGE11 = 384
    self.nstripsGE21 = 768
    self.npadsGE11 = 96
    self.npadsGE21 = 192
