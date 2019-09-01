import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring
from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring

from DQMOffline.Trigger.TrackingMonitoring_cff import * 

DisplacedJetIter2TracksMonitoringHLT = trackingMonHLT.clone()
DisplacedJetIter2TracksMonitoringHLT.FolderName       = 'HLT/EXO/DisplacedJet/Tracking/iter2MergedForBTag'
DisplacedJetIter2TracksMonitoringHLT.TrackProducer    = 'hltIter2MergedForBTag'
DisplacedJetIter2TracksMonitoringHLT.allTrackProducer = 'hltIter2MergedForBTag'
DisplacedJetIter2TracksMonitoringHLT.doEffFromHitPatternVsPU   = cms.bool(False)
DisplacedJetIter2TracksMonitoringHLT.doEffFromHitPatternVsBX   = cms.bool(False)
DisplacedJetIter2TracksMonitoringHLT.doEffFromHitPatternVsLUMI = cms.bool(False)


DisplacedJetIter4TracksMonitoringHLT = trackingMonHLT.clone()
DisplacedJetIter4TracksMonitoringHLT.FolderName       = 'HLT/EXO/DisplacedJet/Tracking/iter4ForDisplaced'
DisplacedJetIter4TracksMonitoringHLT.TrackProducer    = 'hltDisplacedhltIter4PFlowTrackSelectionHighPurity'
DisplacedJetIter4TracksMonitoringHLT.allTrackProducer = 'hltDisplacedhltIter4PFlowTrackSelectionHighPurity'
DisplacedJetIter4TracksMonitoringHLT.doEffFromHitPatternVsPU   = cms.bool(True)
DisplacedJetIter4TracksMonitoringHLT.doEffFromHitPatternVsBX   = cms.bool(False)
DisplacedJetIter4TracksMonitoringHLT.doEffFromHitPatternVsLUMI = cms.bool(False)

trackingMonitorHLTDisplacedJet = cms.Sequence(
     DisplacedJetIter2TracksMonitoringHLT
    +DisplacedJetIter4TracksMonitoringHLT
)


hltHT_HT425_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT425_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT/HT_425/')
hltHT_HT425_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT425_v*")
hltHT_HT425_Prommonitoring.jetSelection_HT = cms.string("pt > 40 && eta < 3.0")

hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT400_DisplacedDijet40_DisplacedTrack')
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT400_DisplacedDijet40_DisplacedTrack_v*")
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>40 && eta<2.0")
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 3.0")


hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack')
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet40_DisplacedTrack_v*")
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>40 && eta<2.0")
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 3.0")


hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT430_DisplacedDijet60_DisplacedTrack')
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet60_DisplacedTrack_v*")
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>60 && eta<2.0")
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 3.0")


hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT500_DisplacedDijet40_DisplacedTrack')
hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT500_DisplacedDijet40_DisplacedTrack_v*")
hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>40 && eta<2.0")
hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection_HT = cms.string("pt > 40 && eta <3.0")


hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT550_DisplacedDijet60_Inclusive')
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT550_DisplacedDijet60_Inclusive_v*")
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection = cms.string("pt>60 && eta<2.0")
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 3.0")


hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT650_DisplacedDijet60_Inclusive')
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT650_DisplacedDijet60_Inclusive_v*")
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection = cms.string("pt>60 && eta<2.0")
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 3.0")


hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT400_DisplacedDijet40_DisplacedTrack')
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.histoPSet.jetptBinning = cms.vdouble(20.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,46.,48.,50.,55.,60.,70.,80.,100.,120.,170.,220.,300.,400.)
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT400_DisplacedDijet40_DisplacedTrack_v*")
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.ispfjettrg = cms.bool(False)
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.iscalojettrg = cms.bool(True)


hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack')
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.histoPSet.jetptBinning = cms.vdouble(20.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,46.,48.,50.,55.,60.,70.,80.,100.,120.,170.,220.,300.,400.)
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet40_DisplacedTrack_v*")
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.ispfjettrg = cms.bool(False)
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.iscalojettrg = cms.bool(True)


hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DisplacedDijet60_DisplacedTrack')
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.histoPSet.jetptBinning = cms.vdouble(20.,26.,30.,35.,40.,45.,50.,52.,53.,54.,56.,58.,60.,62.,64.,66.,68.,70.,72.,75.,80.,100.,120.,170.,220.,300.,400.)
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet60_DisplacedTrack_v*")
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.ispfjettrg = cms.bool(False)
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.iscalojettrg = cms.bool(True)


hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT500_DisplacedDijet40_DisplacedTrack')
hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.histoPSet.jetptBinning = cms.vdouble(20.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,46.,48.,50.,55.,60.,70.,80.,100.,120.,170.,220.,300.,400.)
hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT500_DisplacedDijet40_DisplacedTrack_v*")
hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.ispfjettrg = cms.bool(False)
hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring.iscalojettrg = cms.bool(True)


hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT550_DisplacedDijet60_Inclusive')
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.ptcut = cms.double(20)
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.histoPSet.jetptBinning = cms.vdouble(20.,26.,30.,35.,40.,45.,50.,52.,53.,54.,56.,58.,60.,62.,64.,66.,68.,70.,72.,75.,80.,100.,120.,170.,220.,300.,400.)
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT550_DisplacedDijet60_Inclusive_v*")
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.ispfjettrg = cms.bool(False)
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.iscalojettrg = cms.bool(True)


hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT650_DisplacedDijet60_Inclusive')
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.ptcut = cms.double(20)
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.histoPSet.jetptBinning = cms.vdouble(20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400)
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT650_DisplacedDijet60_Inclusive_v*")
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.ispfjettrg = cms.bool(False)
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.iscalojettrg = cms.bool(True)


exoHLTDisplacedJetmonitoring = cms.Sequence(

 hltHT_HT425_Prommonitoring
+hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring
+hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring
+hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring
+hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring
+hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring
+hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring
)


