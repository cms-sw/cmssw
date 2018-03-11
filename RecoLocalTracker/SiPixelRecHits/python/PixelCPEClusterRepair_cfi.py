import FWCore.ParameterSet.Config as cms

templates2 = cms.ESProducer("PixelCPEClusterRepairESProducer",
    ComponentName = cms.string('PixelCPEClusterRepair'),
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

# This customization will be removed once we get the templates for phase2 pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(templates2,
  LoadTemplatesFromDB = False,
  DoLorentz = False,
)

