import FWCore.ParameterSet.Config as cms

src1 = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalTBWeightsRcd'),
        tag = cms.string('EcalTBWeightsSM')
    ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            tag = cms.string('EcalWeightXtalGroupsSM')
        )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/ECAL/testbeam/pedestal/2006/WEIGHTS/ecalwgt_Default.db'),
)

src2 = cms.ESSource("EcalTrivialConditionRetriever",
    #       Values to get correct noise on RecHit amplitude using 3+5 weights
    EBpedRMSX12 = cms.untracked.double(1.26),
    weightsForTB = cms.untracked.bool(False),
    producedEcalPedestals = cms.untracked.bool(True),
    #       If set true reading optimized weights (3+5 weights) from file 
    getWeightsFromFile = cms.untracked.bool(False),
    producedEcalWeights = cms.untracked.bool(False),
    EEpedRMSX12 = cms.untracked.double(2.87),
    producedEcalIntercalibConstants = cms.untracked.bool(True),
    producedEcalGainRatios = cms.untracked.bool(True),
    producedEcalADCToGeVConstant = cms.untracked.bool(True)
)

getCond = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalTBWeightsRcd'),
        data = cms.vstring('EcalTBWeights')
    ), 
        cms.PSet(
            record = cms.string('EcalWeightXtalGroupsRcd'),
            data = cms.vstring('EcalWeightXtalGroups')
        )),
    verbose = cms.untracked.bool(True)
)


