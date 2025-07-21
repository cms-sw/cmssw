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

# produce a PathActivityToken when run, to indicate that its path is active for the current event
from FWCore.Modules.modules import PathActivityProducer
process.activityProducer = PathActivityProducer()

# try to consume a PathActivityToken: if it is found accept the event, otherwise reject it
from FWCore.Modules.modules import PathActivityFilter
process.actifityFilter = PathActivityFilter(
    producer = 'activityProducer'
)

# select one event out of two and record the path's activity
process.path = cms.Path(
    process.filter +
    process.activityProducer
)

# replicate the activity of the original path
process.replica = cms.Path(
    process.actifityFilter
)

# compare the results of path and replica
from FWCore.Modules.modules import PathStatusFilter
process.test = PathStatusFilter(
    logicalExpression = '(path and replica) or (not path and not replica)'
)

# schedule the comparison
process.testPath = cms.Path(
    process.test
)

# require that the comparison has been successful for every event
from FWCore.Framework.modules import SewerModule
process.require = SewerModule(
    name = cms.string('require'),
    shouldPass = process.maxEvents.input.value(),
    SelectEvents = dict(
        SelectEvents = 'testPath'
    )
)

process.endp = cms.EndPath(
    process.require
)
