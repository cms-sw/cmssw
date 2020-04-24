import FWCore.ParameterSet.Config as cms

candidateChargeBTagComputer = cms.ESProducer("CandidateChargeBTagESProducer",
    useCondDB = cms.bool(False),
    gbrForestLabel = cms.string(""),
    weightFile = cms.FileInPath('RecoBTag/Combined/data/ChargeBTag_4sep_2016.weights.xml.gz'),
    useAdaBoost = cms.bool(True),
    jetChargeExp = cms.double(0.8),
    svChargeExp = cms.double(0.5)
)
