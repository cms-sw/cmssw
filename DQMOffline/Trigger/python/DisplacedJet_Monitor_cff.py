import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring
from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring

from DQMOffline.Trigger.TrackingMonitoring_cff import * 

DisplacedJetIter2TracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/EXO/DisplacedJet/Tracking/iter2MergedForBTag',
    TrackProducer    = 'hltIter2MergedForBTag',
    allTrackProducer = 'hltIter2MergedForBTag',
    doEffFromHitPatternVsPU   = False,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)

DisplacedJetIter4TracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/EXO/DisplacedJet/Tracking/iter4ForDisplaced',
    TrackProducer    = 'hltDisplacedhltIter4PFlowTrackSelectionHighPurity',
    allTrackProducer = 'hltDisplacedhltIter4PFlowTrackSelectionHighPurity',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
trackingMonitorHLTDisplacedJet = cms.Sequence(
     DisplacedJetIter2TracksMonitoringHLT
    +DisplacedJetIter4TracksMonitoringHLT
)

hltHT_HT425_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_425/',
    jetSelection_HT = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT425_v*"])
)

hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT400_DisplacedDijet40_DisplacedTrack',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT400_DisplacedDijet40_DisplacedTrack_v*"])
)

hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet40_DisplacedTrack_v*"])
)

hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT430_DisplacedDijet60_DisplacedTrack',
    jetSelection = "pt>60 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet60_DisplacedTrack_v*"])
)

hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT500_DisplacedDijet40_DisplacedTrack',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT = "pt > 40 && eta <3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT500_DisplacedDijet40_DisplacedTrack_v*"])
)

hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT550_DisplacedDijet60_Inclusive',
    jetSelection = "pt>60 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT550_DisplacedDijet60_Inclusive_v*"])
)

hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT650_DisplacedDijet60_Inclusive',
    jetSelection = "pt>60 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT650_DisplacedDijet60_Inclusive_v*"])
)

hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT400_DisplacedDijet40_DisplacedTrack',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,46.,48.,50.,55.,60.,70.,80.,100.,120.,170.,220.,300.,400.]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT400_DisplacedDijet40_DisplacedTrack_v*"])
)

hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,46.,48.,50.,55.,60.,70.,80.,100.,120.,170.,220.,300.,400.]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet40_DisplacedTrack_v*"])
)

hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DisplacedDijet60_DisplacedTrack',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20.,26.,30.,35.,40.,45.,50.,52.,53.,54.,56.,58.,60.,62.,64.,66.,68.,70.,72.,75.,80.,100.,120.,170.,220.,300.,400.]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet60_DisplacedTrack_v*"])
)

hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT500_DisplacedDijet40_DisplacedTrack',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,46.,48.,50.,55.,60.,70.,80.,100.,120.,170.,220.,300.,400.]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT500_DisplacedDijet40_DisplacedTrack_v*"])
)

hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT550_DisplacedDijet60_Inclusive',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20.,26.,30.,35.,40.,45.,50.,52.,53.,54.,56.,58.,60.,62.,64.,66.,68.,70.,72.,75.,80.,100.,120.,170.,220.,300.,400.]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT550_DisplacedDijet60_Inclusive_v*"])
)

hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT650_DisplacedDijet60_Inclusive',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT650_DisplacedDijet60_Inclusive_v*"])
)

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


