import FWCore.ParameterSet.Config as cms

trackerOnlyConversions = cms.EDProducer('TrackerOnlyConversionProducer',
    AlgorithmName = cms.string('trackerOnly'),
    #src = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("secStep"), cms.InputTag("thStep")),
    src = cms.VInputTag(cms.InputTag("generalTracks")),
    convertedPhotonCollection = cms.string(''), ## or empty

    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),

    HalfwayEta = cms.double(.1),# Track pairing search range on Eta

    #tight cut
    #d0 = cms.double(0.), #d0*charge cut
    #dEtaTrackBC = cms.double(.06), # Track-Basic cluster matching, position diff on eta
    #dPhiTrackBC = cms.double(.6), # Track-Basic cluster matching, position diff on phi
    #EnergyBC = cms.double(1.5), # Track-Basic cluster matching, BC energy lower cut
    #EnergyTotalBC = cms.double(5.), # Track-Basic cluster matching, two BC energy summation cut
    #MaxChi2Left = cms.double(5.), #Track quality
    #MaxChi2Right = cms.double(5.),
    #MinHitsLeft = cms.int32(5),
    #MinHitsRight = cms.int32(2),
    #DeltaCotTheta = cms.double(.02), #Track pair opening angle on R-Z
    #DeltaPhi = cms.double(.2), #Track pair opening angle on X-Y (not a final selection cut)
    #MinApproach = cms.double(-.0), #Track pair min distance at approaching point on X-Y
    
    #loose cut
    d0 = cms.double(0.), #d0*charge cut
    dEtaTrackBC = cms.double(.06), # Track-Basic cluster matching, position diff on eta
    dPhiTrackBC = cms.double(.6), # Track-Basic cluster matching, position diff on phi
    EnergyBC = cms.double(0.9), # Track-Basic cluster matching, BC energy lower cut
    EnergyTotalBC = cms.double(2.), # Track-Basic cluster matching, two BC energy summation cut
    MaxChi2Left = cms.double(5.), #Track quality
    MaxChi2Right = cms.double(5.),
    MinHitsLeft = cms.int32(5),
    MinHitsRight = cms.int32(2),
    DeltaCotTheta = cms.double(.036), #Track pair opening angle on R-Z
    DeltaPhi = cms.double(.2), #Track pair opening angle on X-Y (not a final selection cut)
    MinApproach = cms.double(-.05), #Track pair min distance at approaching point on X-Y

    AllowD0 = cms.bool(True), #Allow d0*charge cut
    AllowTrackBC = cms.bool(True), #Allow to match track-basic cluster
    AllowDeltaCot = cms.bool(True), #Allow pairing using delta cot theta cut
    AllowMinApproach = cms.bool(True), #Allow pairing using min approach cut
    AllowOppCharge = cms.bool(True), #use opposite charge tracks to pair
    AllowSingleLeg = cms.bool(False), #Allow single track conversion
    AllowRightBC = cms.bool(True) #Require second leg matching basic cluster
)
