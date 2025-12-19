import FWCore.ParameterSet.Config as cms

from FWCore.Services.InitRootHandlers_cfi import *

# Tell TFileAdaptor to suppress the statistics summary.  We aren't actually using
# that service and it's summary interferes with the MessageLogger summary.
TFileAdaptor = cms.Service("TFileAdaptor",
    stats = cms.untracked.bool(False)
)
