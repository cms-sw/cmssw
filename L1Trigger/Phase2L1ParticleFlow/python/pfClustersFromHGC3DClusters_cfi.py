import FWCore.ParameterSet.Config as cms

### this below is not ready yet 
pfClustersFromHGC3DClusters = cms.EDProducer("PFClusterProducerFromHGC3DClusters",
   src = cms.InputTag("hgcalTriggerPrimitiveDigiProducer","cluster3D"),
   emOnly = cms.bool(False),
   etMin = cms.double(0.0), 
   corrector  = cms.string(""), # not available yet
   resol = cms.PSet( # dummy numbers for now
           etaBins = cms.vdouble( 3.50),
           offset  = cms.vdouble( 2.50),
           scale   = cms.vdouble( 0.15),
           kind    = cms.string('calo'),
   )
)
