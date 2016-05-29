#
# Author: Ph Gras. CEA/IRFU - Saclay
#

import FWCore.ParameterSet.Config as cms

ecalRawDataRecovery = cms.EDProducer("EcalRawDataRecovery",
                                     inputCollection = cms.InputTag('rawDataCollector'),
                                     outputCollectionLabel = cms.string('fixedFeds')
                                     )
