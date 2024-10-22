import FWCore.ParameterSet.Config as cms
import copy

caloEventShapeVars = cms.EDProducer("EventShapeVarsProducer",
    # name of input collection used for computation of event-shape variables
    # (any collection of objects inheriting from reco::Candidate can be used as input)                               
    src = cms.InputTag("towerMaker"),

    # momentum dependence of sphericity and aplanarity variables and of C and D quantities 
    # (r = 2. corresponds to the conventionally used default, but raises issues of infrared safety in QCD calculations;
    #  see https://arxiv.org/pdf/hep-ph/0603175v2.pdf#page=524 for more details)
    r = cms.double(2.),

    # number of Fox-Wolfram moments to compute
    fwmax = cms.uint32(0),
)

pfEventShapeVars = caloEventShapeVars.clone(
    src = cms.InputTag("pfNoPileUp")
)

produceEventShapeVars = cms.Sequence( caloEventShapeVars * pfEventShapeVars )
