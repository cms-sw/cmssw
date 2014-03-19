import sys,os

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
sys.argv.append( '-b' )
ROOT.gROOT.SetBatch(1)

def enum(*sequential, **named):
  enums = dict(zip(sequential, range(len(sequential))), **named)
  reverse = dict((value, key) for key, value in enums.iteritems())
  enums['reverse_mapping'] = reverse
  return type('Enum', (), enums)

class SimHitPlotter():
  def __init__(self):
    self.inputDir = os.getenv("CMSSW_BASE") + "/src/"
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
    self.sel = [muon,nonMuon,all]
    self.pre = ["Muon","Non muon","All"]
    self.suff = ["_muon","_nonmuon","_all"]
    self.geometry = "custom_GE11_6partitions_v1"

    
class DigiPlotter():
  def __init__(self):
    self.inputDir = os.getenv("CMSSW_BASE") + "/src/"
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

class GEMCSCStubPlotter():
  def __init__(self):
    self.inputDir = os.getenv("CMSSW_BASE") + "/src/"
    self.inputFile = "gem-csc_stub_ana.root"
    self.targetDir = "gem_csc_matching/"
    self.ext = ".png"
    self.analyzer = "GEMCSCAnalyzer"
    self.effSt = "trk_eff_"
    self.stations = enum('ALL','ME11','ME1a','ME1b','ME12','ME13','ME21','ME22','ME31','ME32','ME41','ME42')
    self.stationsToUse = [self.stations.ME11,self.stations.ME1a,self.stations.ME1b,
                          self.stations.ME21,self.stations.ME31,self.stations.ME41]
    self.file = TFile.Open(self.inputDir + self.inputFile)
    self.dirAna = (self.file).Get(self.analyzer)
    self.treeEffSt = []
    for x in self.stationsToUse:
      self.treeEffSt.append(self.dirAna.Get(self.effSt + self.stations.reverse_mapping[x]))
    self.yMin = 0.8
    self.yMax = 1.02
    self.etaMin = 1.5
    self.etaMax = 2.5
    self.pu = 140
