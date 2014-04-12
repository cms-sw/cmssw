import FWCore.ParameterSet.Config as cms

#
# MuonCalIsolationProducer
# -> isolation of muons against the sum of transverse energy depositions in the calorimeter 
#   
MuonCalIsolation = cms.EDProducer("MuonCalIsolationProducer",
    #
    # Source and Swarm collection
    #
    src = cms.InputTag("globalMuons"),
    elements = cms.InputTag("towerMaker"),
    #
    # Propagate the source to the calorimeter:
    # dimensions of target cylinder in cm (full tracker: r=112, z=+/- 270)
    CalRadius = cms.double(112.0),
    dRMax = cms.double(0.3),
    IgnoreMaterial = cms.bool(False), ##Should SteppingHelixPropagator take into account material?

    PropagateToCal = cms.bool(False),
    CalMaxZ = cms.double(270.0), ##cm

    #
    # selection cuts
    #
    dRMin = cms.double(0.0),
    CalMinZ = cms.double(-270.0) ##cm

)


