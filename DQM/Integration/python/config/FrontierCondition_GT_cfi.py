import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from Configuration.AlCa.autoCond import autoCond
GlobalTag.globaltag = autoCond['run3_hlt']

#############################################
#
#              DO NOT REMOVE
#
# This GlobalTag customization is necessary to 
# refresh the online BeamSpot ESProducer inputs
# used by the online DQM clients at every LS
# (as it done in the HLT menu).
##############################################
GlobalTag.toGet = cms.VPSet(
    cms.PSet( record = cms.string( "BeamSpotOnlineLegacyObjectsRcd" ),
              refreshTime = cms.uint64( 1 ),
            ),
    cms.PSet( record = cms.string( "BeamSpotOnlineHLTObjectsRcd" ),
              refreshTime = cms.uint64( 1 )
            )
)
