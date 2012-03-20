
import FWCore.ParameterSet.Config as cms

egHLTOffEleBarrelCuts = cms.PSet (
    #---Morse-----
    #cuts set to WP80
    #cuts=cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:e2x5Over5x5:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksEle:hltIsolEm"),
    cuts=cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:minr9:maxr9:e2x5Over5x5:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksEle:hltIsolEm"),
    #------------
    minEt=cms.double(15),
    minEta=cms.double(0.),
    maxEta=cms.double(1.442),
    maxDEtaIn=cms.double(0.004),#WP80
    maxDPhiIn=cms.double(0.06),#WP80
    maxInvEInvP=cms.double(0.1),
    maxHadem=cms.double(0.04),#WP80
    maxHadEnergy=cms.double(0),
    maxSigmaIEtaIEta=cms.double(0.01),#WP80
    maxSigmaEtaEta=cms.double(0.01),#WP80
    #----Morse------
    #minR9=cms.double(0.9),
    minR9=cms.double(0.0),
    maxR9=cms.double(0.98),
    #---------------
    isolEmConstTerm=cms.double(3),
    isolEmGradTerm=cms.double(0.03),
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(3),
    isolHadGradTerm=cms.double(0.03),
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(7.5),
    isolPtTrksGradTerm=cms.double(0.0),
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(0),#not used
    maxHLTIsolTrksEle = cms.double(0),
    maxHLTIsolTrksEleOverPt = cms.double(0.09),#WP80
    maxHLTIsolTrksEleOverPt2 = cms.double(0),
    maxHLTIsolTrksPho = cms.double(0),
    maxHLTIsolTrksPhoOverPt = cms.double(0),
    maxHLTIsolTrksPhoOverPt2 = cms.double(0),
    maxHLTIsolHad = cms.double(0),
    maxHLTIsolHadOverEt = cms.double(0.10),#WP80
    maxHLTIsolHadOverEt2 = cms.double(0),
    maxHLTIsolEm = cms.double(0),
    maxHLTIsolEmOverEt = cms.double(0.07),#WP80
    maxHLTIsolEmOverEt2 = cms.double(0),

    minCTFTrkOuterRadius=cms.double(0.),
    maxCTFTrkInnerRadius=cms.double(99999),
    minNrCTFTrkHits=cms.int32(7),
    maxNrCTFTrkHitsLost=cms.int32(999),
    maxCTFTrkChi2NDof=cms.double(99999),
    requirePixelHitsIfOuterInOuter=cms.bool(True),

    maxHLTDEtaIn=cms.double(0.008),
    maxHLTDPhiIn=cms.double(0.1),
    maxHLTInvEInvP=cms.double(9999),
    
    )

egHLTOffEleEndcapCuts = cms.PSet (
    #-----Morse-----
    #cuts=cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksEle:hltIsolEm"),
    cuts=cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:minr9:maxr9:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksEle:hltIsolEm"),
    #--------------
    minEt=cms.double(15),
    minEta=cms.double(1.56),
    maxEta=cms.double(2.5),
    maxDEtaIn=cms.double(0.007),#WP80
    maxDPhiIn=cms.double(0.03),#WP80
    maxInvEInvP=cms.double(0.1),
    maxHadem=cms.double(0.025),#WP80
    maxHadEnergy=cms.double(0),
    maxSigmaIEtaIEta=cms.double(0.03),#WP80
    maxSigmaEtaEta=cms.double(0.03),#WP80
    #----Morse------
    #minR9=cms.double(0.9),
    minR9=cms.double(0.0),
    maxR9=cms.double(999.),
    #---------------
    isolEmConstTerm=cms.double(3),
    isolEmGradTerm=cms.double(0.03),
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(3),
    isolHadGradTerm=cms.double(0.03),
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(15),
    isolPtTrksGradTerm=cms.double(0.0),
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(0),#not used
    maxHLTIsolTrksEle = cms.double(0),
    maxHLTIsolTrksEleOverPt = cms.double(0.04),#WP80
    maxHLTIsolTrksEleOverPt2 = cms.double(0),
    maxHLTIsolTrksPho = cms.double(0),
    maxHLTIsolTrksPhoOverPt = cms.double(0),
    maxHLTIsolTrksPhoOverPt2 = cms.double(0),
    maxHLTIsolHad = cms.double(0),
    maxHLTIsolHadOverEt = cms.double(0.025),#WP80
    maxHLTIsolHadOverEt2 = cms.double(0),
    maxHLTIsolEm = cms.double(0),
    maxHLTIsolEmOverEt = cms.double(0.05),#WP80
    maxHLTIsolEmOverEt2 = cms.double(0),

    minCTFTrkOuterRadius=cms.double(0.),
    maxCTFTrkInnerRadius=cms.double(9999.),
    minNrCTFTrkHits=cms.int32(7),
    maxNrCTFTrkHitsLost=cms.int32(999),
    maxCTFTrkChi2NDof=cms.double(99999),
    requirePixelHitsIfOuterInOuter=cms.bool(True),
    
    maxHLTDEtaIn=cms.double(0.008),
    maxHLTDPhiIn=cms.double(0.1),
    maxHLTInvEInvP=cms.double(9999),
    )

egHLTOffEleCuts = cms.PSet(
    barrel = cms.PSet(egHLTOffEleBarrelCuts),
    endcap = cms.PSet(egHLTOffEleEndcapCuts)
    )

egHLTOffEleLooseCuts = cms.PSet(
    barrel = cms.PSet(egHLTOffEleBarrelCuts),
    endcap = cms.PSet(egHLTOffEleEndcapCuts)
    )
