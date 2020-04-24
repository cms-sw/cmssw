import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEfromTemplateESProducer = stripCPEESProducer.clone()
StripCPEfromTemplateESProducerComponentName = cms.string('StripCPEfromTemplate')
StripCPEfromTemplateESProducer.ComponentType = cms.string('StripCPEfromTemplate')
StripCPEfromTemplateESProducer.parameters = cms.PSet(
   UseTemplateReco            = cms.bool(False),
   TemplateRecoSpeed          = cms.int32(0),
   UseStripSplitClusterErrors = cms.bool(False)
)


