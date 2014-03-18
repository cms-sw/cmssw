
# Run only the legacy emulation (pre-LS1 system only).

import FWCore.ParameterSet.Config as cms

#from Configuration.StandardSequences.Geometry_cff import *
from Configuration.Geometry.GeometryIdeal_cff import *
from Configuration.StandardSequences.RawToDigi_Data_cff import *
from L1Trigger.RegionalCaloTrigger.rctDigis_cfi import *
from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *

rctDigis = cms.EDProducer("L1RCTProducer",
                          ecalDigis = cms.VInputTag(cms.InputTag("ecalDigis:EcalTriggerPrimitives")),
                          hcalDigis = cms.VInputTag(cms.InputTag("hcalDigis")),
                          useEcal = cms.bool(True),
                          useHcal = cms.bool(True),  
                          BunchCrossings = cms.vint32(0),
                          getFedsFromOmds = cms.bool(False),
                          #    getFedsFromOmds = cms.bool(True), # ONLY for online DQM!
                          queryDelayInLS = cms.uint32(10),
                          queryIntervalInLS = cms.uint32(100)#,
                          )

digiStep = cms.Sequence(
    # Only do the digitization of objects that we care about
    #RawToDigi
    gctDigis
    #* gtDigis
    * ecalDigis
    * hcalDigis
    )

emulatorStep = cms.Sequence(
    rctDigis
    *l1extraParticles
    )

