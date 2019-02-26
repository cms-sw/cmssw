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

from Alignment.OfflineValidation.overlapValidationPlot import plot

plot(
".oO[datadir]Oo./.oO[PlotsDirName]Oo./BPIX",
.oO[PlottingInstantiation]Oo.
)

"""
