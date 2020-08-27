overlapTemplate = """
# Use compressions settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 207
process.analysis = cms.EDAnalyzer("OverlapValidation",
    usePXB = cms.bool(True),
    usePXF = cms.bool(True),
    useTIB = cms.bool(True),
    useTOB = cms.bool(True),
    useTID = cms.bool(True),
    useTEC = cms.bool(True),
    compressionSettings = cms.untracked.int32(compressionSettings),
    ROUList = cms.vstring('TrackerHitsTIBLowTof',
        'TrackerHitsTIBHighTof',
        'TrackerHitsTOBLowTof',
        'TrackerHitsTOBHighTof'),
    trajectories = cms.InputTag("FinalTrackRefitter"),
    associatePixel = cms.bool(False),
    associateStrip = cms.bool(False),
    associateRecoTracks = cms.bool(False),
    tracks = cms.InputTag("FinalTrackRefitter"),
    barrelOnly = cms.bool(False)
)

"""

overlapValidationSequence = "process.analysis"

overlapPlottingTemplate = """

import os
import ROOT
from Alignment.OfflineValidation.TkAlStyle import TkAlStyle

TkAlStyle.legendheader = ".oO[legendheader]Oo."
TkAlStyle.set(ROOT..oO[publicationstatus]Oo., ROOT..oO[era]Oo., ".oO[customtitle]Oo.", ".oO[customrighttitle]Oo.")

try:
  os.makedirs(".oO[datadir]Oo./.oO[PlotsDirName]Oo./")
except OSError:
  pass
try:
  os.makedirs(".oO[datadir]Oo./.oO[PlotsDirName]Oo./Profiles")
except OSError:
  pass

from Alignment.OfflineValidation.overlapValidationPlot import plot

subdet_ids=[True,True,True,True,True,True]#(BPIX,FPIX,TIB,TID,TOB,TEC)
module_directions=[True,True,True]#(z,r,phi)
overlap_directions=[True,True,True]#(z,r,phi)
profile_directions=[True,True,True,True]#(histogtam,z-profiles,r-profiles,phi-profiles)


plot(".oO[datadir]Oo./.oO[PlotsDirName]Oo./",subdet_ids,module_directions,overlap_directions,profile_directions,.oO[PlottingInstantiation]Oo.)


"""
