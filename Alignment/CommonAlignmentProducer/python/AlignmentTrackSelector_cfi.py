
import FWCore.ParameterSet.Config as cms

AlignmentTrackSelector = cms.EDFilter("AlignmentTrackSelectorModule",
    src = cms.InputTag("generalTracks"),
    filter = cms.bool(False),

    # Settings for the base TrackSelector 	
    # FIXME this should get its own PSet
    applyBasicCuts = cms.bool(True),
    ptMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    pMin = cms.double(0.0),
    pMax = cms.double(9999.0),
    etaMin = cms.double(-2.6),
    etaMax = cms.double(2.6),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    chi2nMax = cms.double(999999.0),
    theCharge = cms.int32(0),   ## -1 neg charge, +1 pos charge, 0 all charges 
    d0Min = cms.double(-999999.0),
    d0Max = cms.double(+999999.0),                                      
    dzMin = cms.double(-999999.0),
    dzMax = cms.double(+999999.0),                                      
    nHitMin = cms.double(0.0),
    nHitMax = cms.double(999.0),
    nLostHitMax = cms.double(999.0),          
    nHitMin2D = cms.uint32(0),
    RorZofFirstHitMin = cms.vdouble(0.0,0.0),
    RorZofFirstHitMax = cms.vdouble(999.0,999.0),
    RorZofLastHitMin = cms.vdouble(0.0,0.0),
    RorZofLastHitMax = cms.vdouble(999.0,999.0),
    countStereoHitAs2D = cms.bool(True),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0),
        inPIXEL = cms.int32(0),
        inTIDplus = cms.int32(0),
        inTIDminus = cms.int32(0),
        inTECplus = cms.int32(0),
        inTECminus = cms.int32(0),
        inFPIXplus = cms.int32(0),
        inFPIXminus = cms.int32(0),
        inENDCAP = cms.int32(0),
        inENDCAPplus = cms.int32(0),
        inENDCAPminus = cms.int32(0),
    ),
    maxHitDiffEndcaps = cms.double(999.0),
    seedOnlyFrom = cms.int32(0),

    applyMultiplicityFilter = cms.bool(False),
    minMultiplicity = cms.int32(1),
    maxMultiplicity = cms.int32(999999),
    multiplicityOnInput = cms.bool(False),

    applyNHighestPt = cms.bool(False),
    nHighestPt = cms.int32(2),

    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    applyIsolationCut = cms.bool(False),
    minHitIsolation = cms.double(0.01),
    applyChargeCheck = cms.bool(False),
    minHitChargeStrip = cms.double(20.0),

    # Settings for the global track selector
    GlobalSelector = cms.PSet(
        #for global muon finding
        applyGlobalMuonFilter = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        maxTrackDeltaR = cms.double(0.001),
        minGlobalMuonCount = cms.int32(1),

        #for isolation Tests
        applyIsolationtest = cms.bool(False),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0), ##GeV
        minJetDeltaR = cms.double(0.2),
        minIsolatedCount = cms.int32(0),

        #for Jet Count
        applyJetCountFilter = cms.bool(False),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        minJetPt = cms.double(40.0), ##GeV
        maxJetCount = cms.int32(3)
    ),

    # Settings for the two Body Decay TrackSelector
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(False),
        daughterMass = cms.double(0.105), ##GeV

        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(15000.0), ##GeV

        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0), ##radian

        minXMass = cms.double(0.0), ##GeV

        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False),

        numberOfCandidates = cms.uint32(1),
        applySecThreshold = cms.bool(False),
        secondThreshold = cms.double(6.0)
    ),
    trackQualities = cms.vstring(), # take all if empty
    iterativeTrackingSteps = cms.vstring(), # take all if empty
    #settings for filtering on the hits taken by the Skim&Prescale workflow
    hitPrescaleMapTag = cms.InputTag(''), # ignore prescale map if empty
    minPrescaledHits = cms.int32(-1)                                  

)


