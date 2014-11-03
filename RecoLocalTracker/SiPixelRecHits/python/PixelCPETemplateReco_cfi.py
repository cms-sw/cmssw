import FWCore.ParameterSet.Config as cms

templates = cms.ESProducer("PixelCPETemplateRecoESProducer",
    ComponentName = cms.string('PixelCPETemplateReco'),
    speed = cms.int32(-2),
    #PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True),
    UseClusterSplitter = cms.bool(False),

    # petar, for clusterProbability() from TTRHs
    ClusterProbComputationFlag = cms.int32(0),
    # gavril
    DoCosmics = cms.bool(False), 
    # The flag to regulate if the LA offset is taken from Alignment 
    # True in Run II for offline RECO
    DoLorentz = cms.bool(True),
 
    LoadTemplatesFromDB = cms.bool(True)

)
