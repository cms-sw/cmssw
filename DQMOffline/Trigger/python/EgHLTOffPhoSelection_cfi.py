import FWCore.ParameterSet.Config as cms

egHLTOffPhoBarrelCuts = cms.PSet (
    cuts=cms.string("et:detEta:hadem:sigmaIEtaIEta:r9:isolEm:isolHad:isolPtTrks"),
    minEt=cms.double(20),
    minEta=cms.double(0.),
    maxEta=cms.double(1.442),
    maxDEtaIn=cms.double(-1),#not used for pho 
    maxDPhiIn=cms.double(-1),#not used for pho 
    maxInvEInvP=cms.double(-1),#not used for pho 
    maxHadem=cms.double(0.05),
    maxSigmaIEtaIEta=cms.double(0.015),
    minR9=cms.double(0.8),
    isolEmConstTerm=cms.double(5),
    isolEmGradTerm=cms.double(0.0),
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(5),
    isolHadGradTerm=cms.double(0.0),
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(9.),
    isolPtTrksGradTerm=cms.double(0.0),
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(4),
    maxHLTIsolTrksEle = cms.double(9999999.0),
    maxHLTIsolTrksPho = cms.double(9999999.0),
    maxHLTIsolHad = cms.double(9999999.0),
    maxHLTIsolHadOverEt = cms.double(9999999.0),
    maxHLTIsolHadOverEt2 = cms.double(9999999.0)
    )

egHLTOffPhoEndcapCuts = cms.PSet (
    cuts=cms.string("et:detEta:hadem:sigmaIEtaIEta:r9:isolEm:isolHad:isolPtTrks:hltIsolHad:hltIsolTrksPho"),
    minEt=cms.double(20),
    minEta=cms.double(1.56),
    maxEta=cms.double(2.5),
    maxDEtaIn=cms.double(-1),#not used for pho 
    maxDPhiIn=cms.double(-1),#not used for pho 
    maxInvEInvP=cms.double(-1),#not used for pho 
    maxHadem=cms.double(0.05),
    maxSigmaIEtaIEta=cms.double(0.015),
    minR9=cms.double(0.8),
    isolEmConstTerm=cms.double(5),
    isolEmGradTerm=cms.double(0.0),
    isolEmGradStart=cms.double(0.),
    isolHadConstTerm=cms.double(5),
    isolHadGradTerm=cms.double(0.0),
    isolHadGradStart=cms.double(0.),
    isolPtTrksConstTerm=cms.double(9),
    isolPtTrksGradTerm=cms.double(0.0),
    isolPtTrksGradStart=cms.double(0.),
    isolNrTrksConstTerm=cms.int32(4),
    maxHLTIsolTrksEle = cms.double(9999999.0),
    maxHLTIsolTrksPho = cms.double(9999999.0),
    maxHLTIsolHad = cms.double(9999999.0),
    maxHLTIsolHadOverEt = cms.double(9999999.0),
    maxHLTIsolHadOverEt2 = cms.double(9999999.0)
    )

egHLTOffPhoCuts =  cms.PSet(
    barrel = cms.PSet(egHLTOffPhoBarrelCuts),
    endcap = cms.PSet(egHLTOffPhoEndcapCuts)
)

egHLTOffPhoLooseCuts =  cms.PSet(
    barrel = cms.PSet(egHLTOffPhoBarrelCuts),
    endcap = cms.PSet(egHLTOffPhoEndcapCuts)
)

