import FWCore.ParameterSet.Config as cms



zmumugammaAnalysis = cms.EDAnalyzer("ZToMuMuGammaAnalyzer",
                                    
    ComponentName = cms.string('zmumugammaAnalysis'),
    analyzerName = cms.string('zmumugammaGedValidation'),
    phoProducer = cms.InputTag('gedPhotons'),
    pfCandidates = cms.InputTag("particleFlow"),
    particleBasedIso = cms.InputTag("particleBasedIsolation","gedPhotons"),                                
    muonProducer = cms.InputTag('muons'),
    barrelRecHitProducer = cms.InputTag('reducedEcalRecHitsEB'),
    endcapRecHitProducer = cms.InputTag('reducedEcalRecHitsEE'),
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD",""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    prescaleFactor = cms.untracked.int32(1),
#
    standAlone = cms.bool(False),
#   Muon Selection
    muonMinPt = cms.double(20.0),
    minPixStripHits = cms.int32(10),
    muonMaxChi2 = cms.double(10.0),
    muonMaxDxy      = cms.double(0.2),
    muonMatches   = cms.int32(2),  
    validPixHits  = cms.int32(1),  
    validMuonHits = cms.int32(1),  
    muonTrackIso   = cms.double(3.0),  
    muonTightEta    = cms.double(2.1),  
#   Dimuon selection                                    
    minMumuInvMass = cms.double(60.0),
    maxMumuInvMass = cms.double(120.0),                                                             
#   Photon selection              
    photonMinEt = cms.double(15.0),
    photonMaxEta = cms.double(2.5),
    photonTrackIso = cms.double(0.9),                                
#   MuMuGamma selection
    nearMuonDr       = cms.double(1.0),
    nearMuonHcalIso  = cms.double(1.0),
    farMuonEcalIso   = cms.double(1.0),
    farMuonTrackIso  = cms.double(3.0),
    farMuonMinPt     = cms.double(15.0),
    minMumuGammaInvMass  = cms.double(75.0),
    maxMumuGammaInvMass  = cms.double(105.0),
#                                    
    isHeavyIon = cms.untracked.bool(False),

    # DBE verbosity
    Verbosity = cms.untracked.int32(0),
                                # 1 provides basic output
                                # 2 provides output of the fill step + 1
                                # 3 provides output of the store step + 2

    useTriggerFiltering = cms.bool(False),
    splitHistosEBEE = cms.bool(True),
    makeProfiles = cms.bool(True),
    use2DHistos = cms.bool(False),
        
    ##### Histogram Ranges and Bins                               

    eBin = cms.int32(150),
    eMin = cms.double(0.0),
    eMax = cms.double(150.0),
                                
    etBin = cms.int32(150),
    etMin = cms.double(0.0),
    etMax = cms.double(150.0),

    sumBin = cms.int32(230),
    sumMin = cms.double(-3.0),
    sumMax = cms.double(20.0),

    etaBin = cms.int32(200),                               
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),

    phiBin = cms.int32(200),
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
    
    r9Bin = cms.int32(110),
    r9Min = cms.double(0.0),
    r9Max = cms.double(1.1),

    hOverEBin = cms.int32(200),
    hOverEMin = cms.double(0),                               
    hOverEMax = cms.double(0.5), 

    numberBin = cms.int32(9),
    numberMin = cms.double(1),                               
    numberMax = cms.double(10), 

    xBin = cms.int32(300),
    xMin = cms.double(-60),                               
    xMax = cms.double(60),
                                
    yBin = cms.int32(300),
    yMin = cms.double(-60),                               
    yMax = cms.double(60),
                                
    rBin = cms.int32(400),
    rMin = cms.double(0),                               
    rMax = cms.double(80),                                

    zBin = cms.int32(400),
    zMin = cms.double(-200),                               
    zMax = cms.double(200),

    dEtaTracksBin = cms.int32(100),
    dEtaTracksMin = cms.double(-0.2),
    dEtaTracksMax = cms.double(0.2),

    dPhiTracksBin = cms.int32(100),
    dPhiTracksMin = cms.double(-0.5),
    dPhiTracksMax = cms.double(0.5),

    sigmaIetaBin = cms.int32(200),
    sigmaIetaMin = cms.double(0.0),
    sigmaIetaMax = cms.double(0.05),

    eOverPBin = cms.int32(100),
    eOverPMin = cms.double(0.0),
    eOverPMax = cms.double(5.0),

    chi2Bin = cms.int32(100),
    chi2Min = cms.double(0.0),
    chi2Max = cms.double(20.0),                                
                                

                                
    OutputFileName = cms.string('DQMOfflinePhotonsAfterFirstStep.root'),


)
