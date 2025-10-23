import FWCore.ParameterSet.Config as cms

process = cms.Process("FALLBACK2")
from FWIO.RNTupleTempInput.modules import RNTupleTempSource
process.source = RNTupleTempSource(
    fileNames = cms.untracked.vstring(
        'file:testNoProcessFallback1.root',
    )
)

from FWCore.Modules.modules import timestudy_SleepingProducer
process.intProducer = timestudy_SleepingProducer( ivalue = 2, eventTimes = [0] )

from FWCore.Modules.modules import EventIDFilter
process.keeper = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,3), cms.EventID(1,1,4), cms.EventID(1,1,7), cms.EventID(1,1,8)]
)

from FWIO.RNTupleTempOutput.modules import RNTupleTempOutputModule
process.out = RNTupleTempOutputModule(
    fileName = 'testNoProcessFallback2.root',
)

process.p = cms.Path(process.keeper + process.intProducer )
process.e = cms.EndPath(process.out)