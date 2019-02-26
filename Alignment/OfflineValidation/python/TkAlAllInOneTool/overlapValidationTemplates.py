overlapTemplate = """
process.analysis = cms.EDAnalyzer("OverlapValidation",
    usePXB = cms.bool(True),
    usePXF = cms.bool(True),
    useTIB = cms.bool(True),
    useTOB = cms.bool(True),
    useTID = cms.bool(True),
    useTEC = cms.bool(True),
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

from Alignment.OfflineValidation.overlapValidationPlot import plot

plot(
".oO[datadir]Oo./.oO[PlotsDirName]Oo./BPIX",
.oO[PlottingInstantiation]Oo.
)

"""
