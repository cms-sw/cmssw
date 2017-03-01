#
# DEPRECATED:  this file is for local testing of HLT configs only...
#

import sys
import FWCore.ParameterSet.Config as cms

print >> sys.stderr, 'L1Trigger/L1TGlobal/python/StableParameters_cff.py is deprecated, please use GlobalParameters_cff.py instead.'

StableParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

#
# This is current HLT setting:
#  - these parameters are all deprecated (execpt NumberPhysTriggers)
#  - real parameters are defaulted
#  - these parameters allowed but ignored
# This lets us roll out new Condition without changing HLT.  They can update at will.

StableParameters = cms.ESProducer( "StableParametersTrivialProducer",
  NumberL1IsoEG = cms.uint32( 4 ),
  NumberL1JetCounts = cms.uint32( 12 ),
  UnitLength = cms.int32( 8 ),
  NumberL1ForJet = cms.uint32( 4 ),
  IfCaloEtaNumberBits = cms.uint32( 4 ),
  IfMuEtaNumberBits = cms.uint32( 6 ),
  NumberL1TauJet = cms.uint32( 4 ),
  NumberL1Mu = cms.uint32( 4 ),
  NumberConditionChips = cms.uint32( 1 ),
  NumberPsbBoards = cms.int32( 7 ),
  NumberL1CenJet = cms.uint32( 4 ),
  NumberPhysTriggers = cms.uint32( 512 ),
  PinsOnConditionChip = cms.uint32( 512 ),
  NumberL1NoIsoEG = cms.uint32( 4 ),
  NumberTechnicalTriggers = cms.uint32( 64 ),
  NumberPhysTriggersExtended = cms.uint32( 64 ),
  WordLength = cms.int32( 64 ),
  OrderConditionChip = cms.vint32( 1 )
)

