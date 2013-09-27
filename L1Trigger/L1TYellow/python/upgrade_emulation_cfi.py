
# Run only the post-LS1 upgrade emulation.

import FWCore.ParameterSet.Config as cms

#from Configuration.StandardSequences.Geometry_cff import *
from Configuration.Geometry.GeometryIdeal_cff import *
#from Configuration.StandardSequences.RawToDigi_Data_cff import *
#from L1Trigger.RegionalCaloTrigger.rctDigis_cfi import *
#from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *

yellowDigis = cms.EDProducer("L1TYellowProducer")

digiStep = cms.Sequence(
#    #RawToDigi
#    gctDigis
    #* gtDigis
#    * ecalDigis
#    * hcalDigis
    yellowDigis
    )



