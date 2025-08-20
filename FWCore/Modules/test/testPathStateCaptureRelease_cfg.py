import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

from FWCore.Modules.modules import EmptySource
process.source = EmptySource()

process.maxEvents.input = 20

# accept one event out of two
from FWCore.Modules.modules import Prescaler
process.filter = Prescaler(
    prescaleFactor = 2,
    prescaleOffset = 0
)

# produce a PathStateToken when run, to indicate that its path is active for the current event
from FWCore.Modules.modules import PathStateCapture
process.pathStateCapture = PathStateCapture()

# try to consume a PathStateToken: if it is found accept the event, otherwise reject it
from FWCore.Modules.modules import PathStateRelease
process.pathStateRelease = PathStateRelease(
    state = 'pathStateCapture'
)

# select one event out of two and record the path's activity
process.captured = cms.Path(
    process.filter +
    process.pathStateCapture
)

# replicate the activity of the original path
process.released = cms.Path(
    process.pathStateRelease
)

# compare the results of the "captured" and "released" paths
from FWCore.Modules.modules import PathStatusFilter
process.compare = PathStatusFilter(
    logicalExpression = '(captured and released) or (not captured and not released)'
)

# schedule the comparison
process.correct = cms.Path(
    process.compare
)

# require that the comparison has been successful for every event
from FWCore.Framework.modules import SewerModule
process.require = SewerModule(
    name = cms.string('require'),
    shouldPass = process.maxEvents.input.value(),
    SelectEvents = dict(
        SelectEvents = 'correct'
    )
)

process.endpath = cms.EndPath(
    process.require
)
