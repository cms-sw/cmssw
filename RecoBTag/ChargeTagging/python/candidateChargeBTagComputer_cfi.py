import FWCore.ParameterSet.Config as cms

candidateChargeBTagComputer = cms.ESProducer("CandidateChargeBTagESProducer",
    useCondDB = cms.bool(False),
    weightFile = cms.FileInPath('RecoBTag/ChargeTagging/data/ChargeBTag_BDT.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(True),
    jetChargeExp = cms.double(0.8),
    svChargeExp = cms.double(0.5)
)
