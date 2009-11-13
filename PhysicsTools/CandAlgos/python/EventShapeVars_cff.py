import FWCore.ParameterSet.Config as cms
import copy

caloEventShapeVars = cms.EDProducer("EventShapeVarsProducer",
    # name of input collection used for computation of event-shape variables
    # (any collection of objects inheriting from reco::Candidate can be used as input)                               
    src = cms.InputTag("towerMaker"),

    # momentum dependence of sphericity and aplanarity variables and of C and D quantities 
    # (r = 2. corresponds to the conventionally used default, but raises issues of infrared safety in QCD calculations;
    #  see http://cepa.fnal.gov/psm/simulation/mcgen/lund/pythia_manual/pythia6.3/pythia6301/node213.html for more details)
    r = cms.double(2.)
)

pfEventShapeVars = copy.deepcopy(caloEventShapeVars)
pfEventShapeVars.src = cms.InputTag("pfNoPileUp")

produceEventShapeVars = cms.Sequence( caloEventShapeVars * pfEventShapeVars )
