import FWCore.ParameterSet.Config as cms

# HLT scalers client. wittich 8/08
# $Id: HLTScalersClient_cfi.py,v 1.6 2010/04/02 20:48:12 wittich Exp $
# $Log: HLTScalersClient_cfi.py,v $
# Revision 1.6  2010/04/02 20:48:12  wittich
# updates to scale entries by received number of FU's
#
# Revision 1.5  2010/03/17 20:56:18  wittich
# Check for good updates based on mergeCount values
# add code for rates normalized per FU
#
# Revision 1.4  2010/02/16 17:04:31  wmtan
# Framework header migrations
#
# Revision 1.3  2009/12/15 20:41:16  wittich
# better hlt scalers client
#
# Revision 1.2  2009/11/04 03:46:01  lorenzo
# added folder param
#
# Revision 1.1  2008/08/22 20:56:56  wittich
# - add client for HLT Scalers
# - Move rate calculation to HLTScalersClient and slim down the
#   filter-farm part of HLTScalers
#
#
hltsClient = cms.EDAnalyzer("HLTScalersClient",
  dqmFolder = cms.untracked.string("HLT/HLTScalers_EvF"),
  rateIntegWindow = cms.untracked.uint32(3),
  processName = cms.string("HLT"),
  debugDump = cms.untracked.bool(False),
  replacePartialUpdates = cms.untracked.bool(True),
  maxFU = cms.untracked.uint32(4704)
)

