import FWCore.ParameterSet.Config as cms

allConversions = cms.EDProducer('ConversionProducer',
    AlgorithmName = cms.string('mixed'),
    #src = cms.VInputTag(cms.InputTag("generalTracks")),
    src = cms.InputTag("gsfGeneralInOutOutInConversionTrackMerger"),
    convertedPhotonCollection = cms.string(''), ## or empty

    bcEndcapCollection = cms.InputTag('particleFlowSuperClusterECAL:particleFlowBasicClusterECALEndcap'),
    bcBarrelCollection = cms.InputTag('particleFlowSuperClusterECAL:particleFlowBasicClusterECALBarrel'),
    scBarrelProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel'),
    scEndcapProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower'),

    primaryVertexProducer = cms.InputTag('offlinePrimaryVerticesWithBS'),

    deltaEta = cms.double(0.4), #track pair search range in eta (applied even in case of preselection bypass)

    HalfwayEta = cms.double(.1),# Track-bc matching search range on Eta
    maxNumOfTrackInPU =  cms.int32(999999),
    maxTrackRho =  cms.double(120.),
    maxTrackZ =  cms.double(300.),                                    
    minSCEt = cms.double(10.0),
    dEtacutForSCmatching = cms.double(0.03),
    dPhicutForSCmatching = cms.double(0.05),                                       
    dEtaTrackBC = cms.double(.2), # Track-Basic cluster matching, position diff on eta
    dPhiTrackBC = cms.double(1.), # Track-Basic cluster matching, position diff on phi
    EnergyBC = cms.double(0.3), # Track-Basic cluster matching, BC energy lower cut
    EnergyTotalBC = cms.double(.3), # Track-Basic cluster matching, two BC energy summation cut
    #tight cuts
    d0 = cms.double(0.), #d0*charge cut
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
    bypassPreselGsf = cms.bool(True), #bypass preselection for gsf + X pairs
    bypassPreselEcal = cms.bool(False), #bypass preselection for ecal-seeded + X pairs
    bypassPreselEcalEcal = cms.bool(True), #bypass preselection for ecal-seeded + ecal-seeded pairs    
    AllowSingleLeg = cms.bool(False), #Allow single track conversion
    AllowRightBC = cms.bool(False) #Require second leg matching basic cluster
)
