import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring
from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring

from DQMOffline.Trigger.TrackingMonitoringcff import * 

DisplacedJetIter2TracksMonitoringHLT = trackingMonHLT.clone()
DisplacedJetIter2TracksMonitoringHLT.FolderName       = 'HLT/EXO/DisplacedJet/Tracking/iter2MergedForBTag'
DisplacedJetIter2TracksMonitoringHLT.TrackProducer    = 'hltIter2MergedForBTag'
DisplacedJetIter2TracksMonitoringHLT.allTrackProducer = 'hltIter2MergedForBTag'

DisplacedJetIter4TracksMonitoringHLT = trackingMonHLT.clone()
DisplacedJetIter2TracksMonitoringHLT.FolderName       = 'HLT/EXO/DisplacedJet/Tracking/iter4ForDisplaced'
DisplacedJetIter4TracksMonitoringHLT.TrackProducer    = 'hltDisplacedhltIter4PFlowTrackSelectionHighPurity'
DisplacedJetIter4TracksMonitoringHLT.allTrackProducer = 'hltDisplacedhltIter4PFlowTrackSelectionHighPurity'

trackingMonitorHLTDisplacedJet = cms.Sequence(
     DisplacedJetIter2TracksMonitoringHLT
    +DisplacedJetIter4TracksMonitoringHLT
)


hltHT_HT425_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT425_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT_425/HT/')
hltHT_HT425_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT425_v*")
hltHT_HT425_Prommonitoring.jetSelection_HT = cms.string("pt > 40 && eta <5.0")

hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT400_DisplacedDijet40_DisplacedTrack/HT/')
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT400_DisplacedDijet40_DisplacedTrack_v*")
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>40 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")


hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack/HT/')
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet40_DisplacedTrack_v*")
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>40 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")


hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT430_DisplacedDijet60_DisplacedTrack/HT/')
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet60_DisplacedTrack_v*")
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>60 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")


hltHT_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT430_DisplacedDijet80_DisplacedTrack/HT/')
hltHT_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet80_DisplacedTrack_v*")
hltHT_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.jetSelection = cms.string("pt>80 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")



hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT_CaloJet_HLT_HT550_DisplacedDijet60_Inclusive/HT/')
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT550_DisplacedDijet60_Inclusive_v*")
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection = cms.string("pt>60 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")

hltHT_HT550_DisplacedDijet80_Inclusive_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT550_DisplacedDijet80_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT_CaloJet_HLT_HT550_DisplacedDijet80_Inclusive/HT/')
hltHT_HT550_DisplacedDijet80_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT550_DisplacedDijet80_Inclusive_v*")
hltHT_HT550_DisplacedDijet80_Inclusive_Prommonitoring.jetSelection = cms.string("pt>80 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT550_DisplacedDijet80_Inclusive_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")



hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT_CaloJet_HLT_HT650_DisplacedDijet60_Inclusive/HT/')
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT650_DisplacedDijet60_Inclusive_v*")
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection = cms.string("pt>60 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")


hltHT_HT650_DisplacedDijet80_Inclusive_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT650_DisplacedDijet80_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT_CaloJet_HLT_HT650_DisplacedDijet80_Inclusive/HT/')
hltHT_HT650_DisplacedDijet80_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT650_DisplacedDijet80_Inclusive_v*")
hltHT_HT650_DisplacedDijet80_Inclusive_Prommonitoring.jetSelection = cms.string("pt>80 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT650_DisplacedDijet80_Inclusive_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")


hltHT_HT750_DisplacedDijet80_Inclusive_Prommonitoring = hltHTmonitoring.clone()
hltHT_HT750_DisplacedDijet80_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HT_CaloJet_HLT_HT750_DisplacedDijet80_Inclusive/HT/')
hltHT_HT750_DisplacedDijet80_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT750_DisplacedDijet80_Inclusive_v*")
hltHT_HT750_DisplacedDijet80_Inclusive_Prommonitoring.jetSelection = cms.string("pt>80 && eta<2.0 && n90>=3 && emEnergyFraction>0.01 && emEnergyFraction<0.99")
hltHT_HT750_DisplacedDijet80_Inclusive_Prommonitoring.jetSelection_HT  = cms.string("pt > 40 && eta < 5.0")


hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT400_DisplacedDijet40_DisplacedTrack/Jet/')
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetptBinning = cms.vdouble(20,26,28,30,32,34,36,38,40,42,44,46,48,50,55,60,70,80,100,120,170,220,300,400)
hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT400_DisplacedDijet40_DisplacedTrack_v*")


hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT430_DisplacedDijet40_DisplacedTrack/Jet/')
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.jetptBinning = cms.vdouble(20,26,28,30,32,34,36,38,40,42,44,46,48,50,55,60,70,80,100,120,170,220,300,400)
hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet40_DisplacedTrack_v*")


hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT430_DisplacedDijet60_DisplacedTrack/Jet/')
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.jetptBinning = cms.vdouble(20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400)
hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet60_DisplacedTrack_v*")


hltJet_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT430_DisplacedDijet80_DisplacedTrack/Jet/')
hltJet_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.ptcut = cms.double(20)
hltJet_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.jetptBinning = cms.vdouble(20,30,40,50,60,65,68,70,72,74,76,78,80,82,84,86,88,90,92,94,100,120,170,220,300,400)
hltJet_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT430_DisplacedDijet80_DisplacedTrack_v*")


hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT550_DisplacedDijet60_Inclusive/Jet/')
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.ptcut = cms.double(20)
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.jetptBinning = cms.vdouble(20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400)
hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT550_DisplacedDijet60_Inclusive_v*")


hltJet_HT550_DisplacedDijet80_Inclusive_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT550_DisplacedDijet80_Inclusive_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT550_DisplacedDijet80_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT550_DisplacedDijet80_Inclusive/Jet/')
hltJet_HT550_DisplacedDijet80_Inclusive_Prommonitoring.ptcut = cms.double(20)
hltJet_HT550_DisplacedDijet80_Inclusive_Prommonitoring.jetptBinning = cms.vdouble(20,30,40,50,60,65,68,70,72,74,76,78,80,82,84,86,88,90,92,94,100,120,170,220,300,400)
hltJet_HT550_DisplacedDijet80_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT550_DisplacedDijet80_Inclusive_v*")


hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT650_DisplacedDijet60_Inclusive/Jet/')
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.ptcut = cms.double(20)
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.jetptBinning = cms.vdouble(20,26,30,35,40,45,50,52,53,54,56,58,60,62,64,66,68,70,72,75,80,100,120,170,220,300,400)
hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT650_DisplacedDijet60_Inclusive_v*")


hltJet_HT650_DisplacedDijet80_Inclusive_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT650_DisplacedDijet80_Inclusive_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT650_DisplacedDijet80_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT650_DisplacedDijet80_Inclusive/Jet/')
hltJet_HT650_DisplacedDijet80_Inclusive_Prommonitoring.ptcut = cms.double(20)
hltJet_HT650_DisplacedDijet80_Inclusive_Prommonitoring.jetptBinning = cms.vdouble(20,30,40,50,60,65,68,70,72,74,76,78,80,82,84,86,88,90,92,94,100,120,170,220,300,400)
hltJet_HT650_DisplacedDijet80_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT650_DisplacedDijet80_Inclusive_v*")


hltJet_HT750_DisplacedDijet80_Inclusive_Prommonitoring = hltJetMETmonitoring.clone()
hltJet_HT750_DisplacedDijet80_Inclusive_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
hltJet_HT750_DisplacedDijet80_Inclusive_Prommonitoring.FolderName = cms.string('HLT/EXO/DisplacedJet/HLT_CaloJet_HT750_DisplacedDijet80_Inclusive/Jet/')
hltJet_HT750_DisplacedDijet80_Inclusive_Prommonitoring.ptcut = cms.double(20)
hltJet_HT750_DisplacedDijet80_Inclusive_Prommonitoring.jetptBinning = cms.vdouble(20,30,40,50,60,65,68,70,72,74,76,78,80,82,84,86,88,90,92,94,100,120,170,220,300,400)
hltJet_HT750_DisplacedDijet80_Inclusive_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_HT750_DisplacedDijet80_Inclusive_v*")


exoHLTDisplacedJetmonitoring = cms.Sequence(

+hltHT_HT425_Prommonitoring
+hltHT_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltHT_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring
+hltHT_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring
+hltHT_HT550_DisplacedDijet60_Inclusive_Prommonitoring
+hltHT_HT550_DisplacedDijet80_Inclusive_Prommonitoring
+hltHT_HT650_DisplacedDijet60_Inclusive_Prommonitoring
+hltHT_HT650_DisplacedDijet80_Inclusive_Prommonitoring
+hltHT_HT750_DisplacedDijet80_Inclusive_Prommonitoring
+hltJet_HT400_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT430_DisplacedDijet40_DisplacedTrack_Prommonitoring
+hltJet_HT430_DisplacedDijet60_DisplacedTrack_Prommonitoring
+hltJet_HT430_DisplacedDijet80_DisplacedTrack_Prommonitoring
+hltJet_HT550_DisplacedDijet60_Inclusive_Prommonitoring
+hltJet_HT550_DisplacedDijet80_Inclusive_Prommonitoring
+hltJet_HT650_DisplacedDijet60_Inclusive_Prommonitoring
+hltJet_HT650_DisplacedDijet80_Inclusive_Prommonitoring
+hltJet_HT750_DisplacedDijet80_Inclusive_Prommonitoring
)


