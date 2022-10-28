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

hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet40_DisplacedTrack_v*"])
)

hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HLT_CaloJet_HT500_DisplacedDijet40_DisplacedTrack',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT = "pt > 40 && eta <3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT500_DisplacedDijet40_DisplacedTrack_v*"])
)

hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT650_DisplacedDijet60_Inclusive',
    jetSelection = "pt>60 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT650_DisplacedDijet60_Inclusive_v*"])
)

hltHT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5',
    jetSelection = "pt>30 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_v*"])
)

hltHT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose',
    jetSelection = "pt>30 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v*"])
)

hltHT_HT430_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT430_DelayedJet40_SingleDelay1nsTrackless',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v*"])
)

hltHT_HT430_DelayedJet40_SingleDelay2nsInclusive_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT430_DelayedJet40_SingleDelay2nsInclusive',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v*"])
)

hltHT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v*"])
)

hltHT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive',
    jetSelection = "pt>60 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_v*"])
)

hltHT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_v*"])
)

hltHT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless',
    jetSelection = "pt>40 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_v*"])
)

hltHT_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedJet/HT/HT_CaloJet_HLT_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5',
    jetSelection = "pt>30 && eta<2.0",
    jetSelection_HT  = "pt > 40 && eta < 3.0",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_v*"])
)


#################
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,46.,48.,50.,55.,60.,70.,80.,100.,120.,170.,220.,300.,400.]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet40_DisplacedTrack_v*"])
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

hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT650_DisplacedDijet60_Inclusive',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT650_DisplacedDijet60_Inclusive_v*"])
)

hltJet_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_v*"])
)

hltJet_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v*"])
)

hltJet_HT430_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DelayedJet40_SingleDelay1nsTrackless',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v*"])
)

hltJet_HT430_DelayedJet40_SingleDelay2nsInclusive_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT430_DelayedJet40_SingleDelay2nsInclusive',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v*"])
)

hltJet_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v*"])
)

hltJet_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_v*"])
)

hltJet_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_v*"])
)

hltJet_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_v*"])
)

hltJet_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring = hltJetMETmonitoring.clone(
    jetSrc = "ak4CaloJets",
    FolderName = 'HLT/EXO/DisplacedJet/Jet/HLT_CaloJet_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5',
    ptcut = 20,
    ispfjettrg = False,
    iscalojettrg = True,
    histoPSet = dict(jetptBinning = [20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_v*"])
)

exoHLTDisplacedJetmonitoring = cms.Sequence(
 hltHT_HT425_Prommonitoring
+hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring
+hltHT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring
+hltHT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_Prommonitoring
+hltHT_HT430_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring
+hltHT_HT430_DelayedJet40_SingleDelay2nsInclusive_Prommonitoring
+hltHT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_Prommonitoring
+hltHT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring
+hltHT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_Prommonitoring
+hltHT_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring

+hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT500_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring
+hltJet_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring
+hltJet_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_Prommonitoring
+hltJet_HT430_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring
+hltJet_HT430_DelayedJet40_SingleDelay2nsInclusive_Prommonitoring
+hltJet_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_Prommonitoring
+hltJet_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_Prommonitoring
+hltJet_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_Prommonitoring
+hltJet_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_Prommonitoring
)


