import FWCore.ParameterSet.Config as cms

from FWCore.Services.InitRootHandlers_cfi import *

# Tell AdaptorConfig to suppress the statistics summary.  We aren't actually using
# that service and it's summary interferes with the MessageLogger summary.
AdaptorConfig = cms.Service("AdaptorConfig",
    stats = cms.untracked.bool(False)
)
# foo bar baz
# USKqYNmV4ul0R
# fRIz4Hzs5WmWY
