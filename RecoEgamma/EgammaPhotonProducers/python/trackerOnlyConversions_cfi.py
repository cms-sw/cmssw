import FWCore.ParameterSet.Config as cms

trackerOnlyConversions = cms.EDProducer('TrackerOnlyConversionProducer',
    AlgorithmName = cms.string('trackerOnly'),
    #src = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("secStep"), cms.InputTag("thStep")),
    #src = cms.VInputTag(cms.InputTag("generalTracks")),
    #src = cms.InputTag("generalConversionTrackProducer"),
    src = cms.InputTag("gsfGeneralInOutOutInConversionTrackMerger"),
    convertedPhotonCollection = cms.string(''), ## or empty

    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),

    primaryVertexProducer = cms.string('offlinePrimaryVerticesWithBS'),

    HalfwayEta = cms.double(.1),# Track pairing search range on Eta
    maxNumOfTrackInPU =  cms.int32(999999),

    #tight cuts
    d0 = cms.double(0.), #d0*charge cut
    dEtaTrackBC = cms.double(.2), # Track-Basic cluster matching, position diff on eta
    dPhiTrackBC = cms.double(1.), # Track-Basic cluster matching, position diff on phi
    EnergyBC = cms.double(0.3), # Track-Basic cluster matching, BC energy lower cut
    EnergyTotalBC = cms.double(.3), # Track-Basic cluster matching, two BC energy summation cut
    MaxChi2Left = cms.double(10.), #Track quality
    MaxChi2Right = cms.double(10.),
    MinHitsLeft = cms.int32(4),
    MinHitsRight = cms.int32(2),
    DeltaCotTheta = cms.double(0.1), #Track pair opening angle on R-Z
    DeltaPhi = cms.double(.2), #Track pair opening angle on X-Y (not a final selection cut)
    vtxChi2 = cms.double(0.0005),
    MinApproachLow = cms.double(-.25), #Track pair min distance at approaching point on X-Y      
    MinApproachHigh = cms.double(1.0), #Track pair min distance at approaching point on X-Y
    rCut = cms.double(2.0),#analytical track cross point
    dz = cms.double(5.0),#track pair inner position difference

# kinematic vertex fit parameters
    maxDelta = cms.double(0.01),#delta of parameters
    maxReducedChiSq = cms.double(225.),#maximum chi^2 per degree of freedom before fit is terminated
    minChiSqImprovement = cms.double(50.),#threshold for "significant improvement" in the fit termination logic
    maxNbrOfIterations = cms.int32(40),#maximum number of convergence iterations

    UsePvtx = cms.bool(True),
    
    AllowD0 = cms.bool(True), #Allow d0*charge cut
    AllowDeltaPhi = cms.bool(False),
    AllowTrackBC = cms.bool(False), #Allow to match track-basic cluster
    AllowDeltaCot = cms.bool(True), #Allow pairing using delta cot theta cut
    AllowMinApproach = cms.bool(True), #Allow pairing using min approach cut
    AllowOppCharge = cms.bool(True), #use opposite charge tracks to pair
    AllowVertex = cms.bool(True),
    AllowSingleLeg = cms.bool(False), #Allow single track conversion
    AllowRightBC = cms.bool(False) #Require second leg matching basic cluster
)
