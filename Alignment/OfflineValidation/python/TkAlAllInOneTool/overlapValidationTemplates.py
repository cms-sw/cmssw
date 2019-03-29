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
detNameList = ("BPIX", "FPIX", "TIB", "TID", "TOB", "TEC")
for subdetid in (1,2,3,4,5,6):
	for moduledirection in ("phi","z","r"):
		for overlapdirection in ("phi","z","r"):
			if subdetid == 1 and (moduledirection == "r" or overlapdirection =="r"):
				continue
			if subdetid == 2 and (moduledirection == "z" or overlapdirection == "z"):
				continue
			if (subdetid == 3 or subdetid == 5) and (overlapdirection != "phi" or moduledirection == "r"):
				continue
			if (subdetid == 4 or subdetid == 6) and (overlapdirection != "phi" or moduledirection == "z"):
                                continue
			plot(".oO[datadir]Oo./.oO[PlotsDirName]Oo./{0}_{1}_{2}".format(moduledirection,overlapdirection,detNameList[subdetid-1]),subdetid,moduledirection,overlapdirection,None, .oO[PlottingInstantiation]Oo.)

plot(".oO[datadir]Oo./.oO[PlotsDirName]Oo./profile_phi_phi_TOB_z", 5, "phi", "phi", "z", .oO[PlottingInstantiation]Oo.)


"""
