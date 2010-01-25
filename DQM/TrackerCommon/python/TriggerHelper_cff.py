import FWCore.ParameterSet.Config as cms

# "hltInputTag":
# The inout tag has to contain also the process name.
# Removing this parameter completly switches the filter off.
# "hltPaths":
# The filter has no effect, if the vector is empty.
# Paths can be negated by prepending a '~'.
# "andOr":
# False = AND, True = OR
# "errorReply":
#

SiStripHltFilter_SiStripMonitorHardware = cms.PSet(
    hltInputTag = cms.InputTag( "TriggerResults::HLT" ),
    hltPaths    = cms.vstring(),
    andOr       = cms.bool( False ),
    errorReply  = cms.bool( True )
)

SiStripHltFilter_SiStripMonitorDigi = cms.PSet(
    hltInputTag = cms.InputTag( "TriggerResults::HLT" ),
    hltPaths    = cms.vstring(),
    andOr       = cms.bool( False ),
    errorReply  = cms.bool( True )
)

SiStripHltFilter_SiStripMonitorCluster = cms.PSet(
    hltInputTag = cms.InputTag( "TriggerResults::HLT" ),
    hltPaths    = cms.vstring(),
    andOr       = cms.bool( False ),
    errorReply  = cms.bool( True )
)

SiStripHltFilter_SiStripMonitorTrack = cms.PSet(
    hltInputTag = cms.InputTag( "TriggerResults::HLT" ),
    hltPaths    = cms.vstring( 'HLT_PhysicsDeclared'
                             ),
    andOr       = cms.bool( False ),
    errorReply  = cms.bool( True )
)

SiStripHltFilter_TrackerMonitorTrack = cms.PSet(
    hltInputTag = cms.InputTag( "TriggerResults::HLT" ),
    hltPaths    = cms.vstring( 'HLT_PhysicsDeclared'
                             ),
    andOr       = cms.bool( False ),
    errorReply  = cms.bool( True )
)

SiStripHltFilter_TrackingMonitor = cms.PSet(
    hltInputTag = cms.InputTag( "TriggerResults::HLT" ),
    hltPaths    = cms.vstring( 'HLT_PhysicsDeclared'
                             ),
    andOr       = cms.bool( False ),
    errorReply  = cms.bool( True )
)
