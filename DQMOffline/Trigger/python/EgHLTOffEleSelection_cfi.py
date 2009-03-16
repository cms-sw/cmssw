
import FWCore.ParameterSet.Config as cms

egHLTOffEleBarrelCuts = cms.PSet (
    cuts=cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:e2x5Over5x5:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksEle"),
    minEt=cms.double(15),
    minEta=cms.double(0.),
    maxEta=cms.double(1.442),
    maxDEtaIn=cms.double(0.1),
    maxDPhiIn=cms.double(0.1),
    maxInvEInvP=cms.double(0.1),
    maxHadem=cms.double(0.05),
    maxSigmaIEtaIEta=cms.double(0.015),
    minR9=cms.double(0.9),
    isolEmConstTerm=cms.double(3),
    isolEmGradTerm=cms.double(0.02),
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(3),
    isolHadGradTerm=cms.double(0.02),
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(3),
    isolPtTrksGradTerm=cms.double(0.02),
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(0),#not used
    maxHLTIsolTrksEle = cms.double(9999999.0),
    maxHLTIsolTrksPho = cms.double(9999999.0),
    maxHLTIsolHad = cms.double(9999999.0),
    maxHLTIsolHadOverEt = cms.double(9999999.0),
    maxHLTIsolHadOverEt2 = cms.double(9999999.0)
    )

egHLTOffEleEndcapCuts = cms.PSet (
    cuts=cms.string("et:detEta:dEtaIn:dPhiIn:hadem:sigmaIEtaIEta:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksEle"),
    minEt=cms.double(15),
    minEta=cms.double(1.56),
    maxEta=cms.double(2.5),
    maxDEtaIn=cms.double(0.1),
    maxDPhiIn=cms.double(0.1),
    maxInvEInvP=cms.double(0.1),
    maxHadem=cms.double(0.05),    
    maxSigmaIEtaIEta=cms.double(0.0275),
    minR9=cms.double(0.9),
    isolEmConstTerm=cms.double(3),
    isolEmGradTerm=cms.double(0.02),
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(3),
    isolHadGradTerm=cms.double(0.02),
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(3),
    isolPtTrksGradTerm=cms.double(0.02),
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(0),#not used
    maxHLTIsolTrksEle = cms.double(9999999.0),
    maxHLTIsolTrksPho = cms.double(9999999.0),
    maxHLTIsolHad = cms.double(9999999.0),
    maxHLTIsolHadOverEt = cms.double(9999999.0),
    maxHLTIsolHadOverEt2 = cms.double(9999999.0)
    )

egHLTOffEleCuts = cms.PSet(
    barrel = cms.PSet(egHLTOffEleBarrelCuts),
    endcap = cms.PSet(egHLTOffEleEndcapCuts)
    )

egHLTOffEleLooseCuts = cms.PSet(
    barrel = cms.PSet(egHLTOffEleBarrelCuts),
    endcap = cms.PSet(egHLTOffEleEndcapCuts)
    )
