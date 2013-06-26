#
# Author: Ph Gras. CEA/IRFU - Saclay
#

import FWCore.ParameterSet.Config as cms

dumpRaw = cms.EDAnalyzer("EcalDumpRaw",
                         amplCut = cms.untracked.double(5.),
                         dump = cms.untracked.bool(True),
                         dumpAdc = cms.untracked.bool(True),
                         l1aHistory = cms.untracked.bool(False),
                         maxEvt  = cms.untracked.int32(10000),
                         profileFedId = cms.untracked.int32(0),
                         profileRuId = cms.untracked.int32(1),
                         l1aMinX = cms.untracked.int32(1),
                         l1aMaxX = cms.untracked.int32(601),
                         verbosity = cms.untracked.int32(0),
                         writeDCC = cms.untracked.bool(False),
                         # fed_id: EE- is 601-609,  EB is 610-645,  EE- is 646-654
                         # when using 'single sm' fed corresponds to construction number
                         beg_fed_id = cms.untracked.int32(601),
                         end_fed_id = cms.untracked.int32(654),

                         # events as counted in the order they are written to file
                         first_event = cms.untracked.int32(0),

                         # events as counted in the order they are written to file
                         #last_event = cms.untracked.int32(99999),                         

                         #     untracked int32 last_event  = $((nskip+nevts))
                         filename = cms.untracked.string('dump.bin'),

                         # If non empty only listed events will be processed:
                         eventList = cms.untracked.vuint32(),
                         fedRawDataCollectionTag = cms.InputTag('rawDataCollector'),
                         l1AcceptBunchCrossingCollectionTag = cms.InputTag('scalersRawToDigi')
                         )
