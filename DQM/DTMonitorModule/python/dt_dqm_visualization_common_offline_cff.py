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
from Configuration.StandardSequences.MagneticField_0T_cff import *

dt1DRecHits.dtDigiLabel = 'dtunpacker'
#DTLinearDriftAlgo_CosmicData.recAlgoConfig.tTrigModeConfig.kFactor = -0.7


from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
#GlobalTag.globaltag = "CRZT210_V1::All" # or "IDEAL_V2::All" or... 
es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
GlobalTag.globaltag = "CRAFT_V3P::All"



from VisFramework.VisFrameworkBase.VisConfigurationService_cff import *
from VisFramework.VisApplication.iguana_dt_cfi import *

VisConfigurationService.Views = cms.untracked.vstring('RPhi Window', 'RZ Window')
#VisConfigurationService.Views = cms.untracked.vstring('RPhi Window', 'RZ Window','3D Window')
VisConfigurationService.VisAutoStart = cms.untracked.bool(True)
VisConfigurationService.VisCaloAnnotation = cms.untracked.bool(False)
VisConfigurationService.VisExceptionMessage = cms.untracked.string("none")
VisConfigurationService.EnabledTwigs = cms.untracked.vstring('/Objects/CMS Event and Detector/Reco Detector/DTs',
                                                             '/Objects/CMS Event and Detector/Muon DT Event/DT Digis',
                                                             '/Objects/CMS Event and Detector/Muon DT Event/DT 4DSegment',
                                                             '/Objects/CMS Event and Detector/Muon/Barrel/Drift Tubes/Drift Tubes Superlayers',
                                                             '/Objects/CMS Event and Detector/Beampipe',
                                                             '/Objects/CMS Event and Detector/Magnet',
                                                             '/Objects/Event Collections/Run and Event Number'
                                                             )
VisConfigurationService.ContentProxies = cms.untracked.vstring('Framework/EventSetup',
                                                               'Reco/MuonDT')
VisConfigurationService.VisPhiDelta = cms.untracked.double(0.25)

# process.VisConfigurationService.ContentProxies = ('Framework/EventSetup',
#                                                    'Reco/MuonDT',
#                                                    'Reco/MuonCSC',
#                                                    'Reco/MuonRPC',
#                                                    'Reco/Muon',
#                                                    'Reco/Detector')


reco = cms.Sequence(dtunpacker + dt1DRecHits + dt4DSegments)
