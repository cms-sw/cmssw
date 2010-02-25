import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# # the source
# source = cms.Source("NewEventStreamFileReader",
#                      fileNames = cms.untracked.vstring(
#      'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0001.A.storageManager.00.0000.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0003.A.storageManager.00.0000.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0003.A.storageManager.00.0001.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0003.A.storageManager.00.0002.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0004.A.storageManager.00.0000.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0004.A.storageManager.00.0001.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0004.A.storageManager.00.0002.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0004.A.storageManager.00.0003.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0004.A.storageManager.00.0004.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0004.A.storageManager.00.0005.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0004.A.storageManager.00.0006.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0007.A.storageManager.00.0000.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0007.A.storageManager.00.0001.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0007.A.storageManager.00.0002.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0007.A.storageManager.00.0003.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0007.A.storageManager.00.0004.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0007.A.storageManager.00.0005.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0008.A.storageManager.00.0000.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0008.A.storageManager.00.0001.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0008.A.storageManager.00.0002.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0008.A.storageManager.00.0003.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0009.A.storageManager.00.0000.dat',
#  'file:/localdatadisk/DTDQM/GlobalCRAFT1.00071092.0165.A.storageManager.00.0006.dat'
#      )
# )

source = cms.Source("EventStreamHttpReader",
      #sourceURL = cms.string('http://srv-c2d05-14.cms:22100/urn:xdaq-application:lid=30'),
      #sourceURL = cms.string('http://srv-c2d04-30:50082/urn:xdaq-application:lid=29'),
      sourceURL = cms.string('http://cmsdisk1.cms:33100/urn:xdaq-application:lid=50'),
      consumerPriority = cms.untracked.string('normal'),
      max_event_size = cms.int32(7000000),
      consumerName = cms.untracked.string('DQM Source'),
      SelectHLTOutput = cms.untracked.string('hltOutputDQM'),
      max_queue_depth = cms.int32(5),
      maxEventRequestRate = cms.untracked.double(15.0),
      SelectEvents = cms.untracked.PSet(
          SelectEvents = cms.vstring('*')
      ),
      headerRetryInterval = cms.untracked.int32(3)
  )


