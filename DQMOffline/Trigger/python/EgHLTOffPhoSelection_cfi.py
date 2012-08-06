import FWCore.ParameterSet.Config as cms

egHLTOffPhoBarrelCuts = cms.PSet (
    #----Morse------
    #cuts=cms.string("et:detEta:hadem:sigmaIEtaIEta:r9:isolEm:isolHad:isolPtTrks"),
    cuts=cms.string("et:detEta:hadem:sigmaIEtaIEta:minr9:maxr9:isolEm:isolHad:isolPtTrks"),
    #------
    minEt=cms.double(20),
    minEta=cms.double(0.),
    maxEta=cms.double(1.442),
    maxDEtaIn=cms.double(-1),#not used for pho 
    maxDPhiIn=cms.double(-1),#not used for pho 
    maxInvEInvP=cms.double(-1),#not used for pho 
    #maxHadem=cms.double(0.04),
    maxHadem=cms.double(0.05),#CaloIdVTIsoT
    maxHadEnergy=cms.double(0),
    maxSigmaIEtaIEta=cms.double(0.011),#CaloIdVTIsoT
    maxSigmaEtaEta=cms.double(0.011),#CaloIdVTIsoT
    #----Morse------
    #minR9=cms.double(0.8),
    minR9= cms.double(0.0),
    maxR9=cms.double(0.98),
    #---------------
    isolEmConstTerm=cms.double(5.),#CaloIdVTIsoT
    isolEmGradTerm=cms.double(0.012),#CaloIdVTIsoT
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(3.),#CaloIdVTIsoT
    isolHadGradTerm=cms.double(0.005),#CaloIdVTIsoT
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(3.),#CaloIdVTIsoT
    isolPtTrksGradTerm=cms.double(0.002),#CaloIdVTIsoT
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(4),
    maxHLTIsolTrksEle = cms.double(0),
    maxHLTIsolTrksEleOverPt = cms.double(0),
    maxHLTIsolTrksEleOverPt2 = cms.double(0),
    maxHLTIsolTrksPho = cms.double(0),
    maxHLTIsolTrksPhoOverPt = cms.double(0),
    maxHLTIsolTrksPhoOverPt2 = cms.double(0),
    maxHLTIsolHad = cms.double(0),
    maxHLTIsolHadOverEt = cms.double(0),
    maxHLTIsolHadOverEt2 = cms.double(0),
    maxHLTIsolEm = cms.double(0),
    maxHLTIsolEmOverEt = cms.double(0),
    maxHLTIsolEmOverEt2 = cms.double(0),

    #the rest of the cuts are track cuts which always fail for photon
    minCTFTrkOuterRadius=cms.double(40.),
    maxCTFTrkInnerRadius=cms.double(9.),
    minNrCTFTrkHits=cms.int32(5),
    maxNrCTFTrkHitsLost=cms.int32(0),
    maxCTFTrkChi2NDof=cms.double(99999.),
    requirePixelHitsIfOuterInOuter=cms.bool(True),
    maxHLTDEtaIn=cms.double(0.1),
    maxHLTDPhiIn=cms.double(0.1),
    maxHLTInvEInvP=cms.double(0.1),
   
    )

egHLTOffPhoEndcapCuts = cms.PSet (
    #----Morse-----
    #cuts=cms.string("et:detEta:hadem:sigmaIEtaIEta:r9:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksPho"),
    cuts=cms.string("et:detEta:hadem:sigmaIEtaIEta:minr9:maxr9:isolEm:isolHad:isolPtTrks"),
    #------------
    minEt=cms.double(20),
    minEta=cms.double(1.56),
    maxEta=cms.double(2.5),
    maxDEtaIn=cms.double(-1),#not used for pho 
    maxDPhiIn=cms.double(-1),#not used for pho 
    maxInvEInvP=cms.double(-1),#not used for pho 
    maxHadem=cms.double(0.05),#CaloIdVTIsoT
    maxHadEnergy=cms.double(0),
    maxSigmaIEtaIEta=cms.double(0.031),#CaloIdVTIsoT
    maxSigmaEtaEta=cms.double(0.031),#CaloIdVTIsoT
    #----Morse------
    #minR9=cms.double(0.8),
    minR9=cms.double(0.0),
    maxR9=cms.double(999.),
    #---------------
    isolEmConstTerm=cms.double(5.),#CaloIdVTIsoT
    isolEmGradTerm=cms.double(0.012),#CaloIdVTIsoT
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(3.),#CaloIdVTIsoT
    isolHadGradTerm=cms.double(0.005),#CaloIdVTIsoT
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(3.),#CaloIdVTIsoT
    isolPtTrksGradTerm=cms.double(0.002),#CaloIdVTIsoT
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(4),
    maxHLTIsolTrksEle = cms.double(0),#not used for pho 
    maxHLTIsolTrksEleOverPt = cms.double(0),#not used for pho 
    maxHLTIsolTrksEleOverPt2 = cms.double(0),#not  used for pho 
    maxHLTIsolTrksPho = cms.double(0),
    maxHLTIsolTrksPhoOverPt = cms.double(0),
    maxHLTIsolTrksPhoOverPt2 = cms.double(0),
    maxHLTIsolHad = cms.double(0),
    maxHLTIsolHadOverEt = cms.double(0),
    maxHLTIsolHadOverEt2 = cms.double(0),
    maxHLTIsolEm = cms.double(0),
    maxHLTIsolEmOverEt = cms.double(0),
    maxHLTIsolEmOverEt2 = cms.double(0),

    #the rest of the cuts are track cuts which always fail for photon
    minCTFTrkOuterRadius=cms.double(40.),
    maxCTFTrkInnerRadius=cms.double(9.),
    minNrCTFTrkHits=cms.int32(5),
    maxNrCTFTrkHitsLost=cms.int32(0),
    maxCTFTrkChi2NDof=cms.double(99999.),
    requirePixelHitsIfOuterInOuter=cms.bool(True),
    maxHLTDEtaIn=cms.double(0.1),
    maxHLTDPhiIn=cms.double(0.1),
    maxHLTInvEInvP=cms.double(0.1),
    
    )

egHLTOffPhoCuts =  cms.PSet(
    barrel = cms.PSet(egHLTOffPhoBarrelCuts),
    endcap = cms.PSet(egHLTOffPhoEndcapCuts)
)

egHLTOffPhoLooseCuts =  cms.PSet(
    barrel = cms.PSet(egHLTOffPhoBarrelCuts),
    endcap = cms.PSet(egHLTOffPhoEndcapCuts)
)
