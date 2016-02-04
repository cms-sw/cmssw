import FWCore.ParameterSet.Config as cms

# filter on trigger type
calibrationEventsFilter = cms.EDFilter("TriggerTypeFilter",
                                       InputLabel = cms.string('source'),
                                       TriggerFedId = cms.int32(812),
                                       # 1=Physics, 2=Calibration, 3=Random, 4=Technical
                                       SelectedTriggerType = cms.int32(1) 
                                       )

# DT digitization and reconstruction
dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    useStandardFEDid = cms.untracked.bool(True),
    fedbyType = cms.untracked.bool(True),
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    )
)

from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.ReconstructionCosmics_cff import *
#from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff import *
dt1DRecHits.dtDigiLabel = 'dtunpacker'

# switch on code for t0 correction
#dt2DSegments.Reco2DAlgoConfig.performT0SegCorrection = True
#dt2DSegments.Reco2DAlgoConfig.T0_hit_resolution = cms.untracked.double(0.0250)

#dt4DSegments.Reco4DAlgoConfig.performT0SegCorrection = True
#dt4DSegments.Reco4DAlgoConfig.T0_hit_resolution = cms.untracked.double(0.0250)
#dt4DSegments.Reco4DAlgoConfig.T0SegCorrectionDebug = True

#DTLinearDriftFromDBAlgo_CosmicData.recAlgoConfig.tTrigModeConfig.kFactor = 0.0

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
#GlobalTag.globaltag = "CRZT210_V1::All" # or "IDEAL_V2::All" or... 
#es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
GlobalTag.globaltag = "CRAFT_V3P::All"


reco = cms.Sequence(dtunpacker + dt1DRecHits + dt2DSegments + dt4DSegments)
