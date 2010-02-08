import FWCore.ParameterSet.Config as cms

# HLT scalers client. wittich 8/08
# $Id: HLTScalers_cfi.py,v 1.3 2008/08/15 15:44:01 wteo Exp $
# $Log: HLTScalers_cfi.py,v $
#
hltsClient = cms.EDFilter("HLTScalersClient",
  # no configuration yet
)

