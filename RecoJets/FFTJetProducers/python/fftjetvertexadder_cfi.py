import FWCore.ParameterSet.Config as cms

fftjetVertexAdder = cms.EDProducer(
    "FFTJetVertexAdder",
    #
    # Label for the beam spot info
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    #
    # Label for an existing collection of primary vertices
    existingVerticesLabel = cms.InputTag("offlinePrimaryVertices"),
    #
    # Label for the output collection
    outputLabel = cms.string("FFTJetFudgedVertices"),
    #
    # Do we want to use the beam spot info from the event data 
    #in order to generate the vertices?
    useBeamSpot = cms.bool(True),
    #
    # Do we want to an existing collection (labeled by "existingVerticesLabel"
    # defined above) to the fake vertices?
    addExistingVertices = cms.bool(False),
    #
    # If we are not using the beam spot, what would be the average
    # position of the generated vertices?
    fixedX = cms.double(0.0),
    fixedY = cms.double(0.0),
    fixedZ = cms.double(0.0),
    #
    # If we are not using the beam spot, what would be the vertex spread?
    sigmaX = cms.double(0.0014),
    sigmaY = cms.double(0.0014),
    sigmaZ = cms.double(6.0),
    #
    # Parameters of the vertex to generate (these are not varied)
    nDof = cms.double(10.0),
    chi2 = cms.double(10.0),
    errX = cms.double(0.001),
    errY = cms.double(0.001),
    errZ = cms.double(0.01),
    #
    # How many fake vertices should we make?
    nVerticesToMake = cms.uint32(1)
)
