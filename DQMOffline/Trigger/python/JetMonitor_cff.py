import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring

### HLT_PFJet Triggers ###
# HLT_PFJet450
PFJet450_Prommonitoring = hltJetMETmonitoring.clone()
PFJet450_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet450/')
PFJet450_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  112 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 1120.),
)
PFJet450_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet450_v*")

# HLT_PFJet40
PFJet40_Prommonitoring = hltJetMETmonitoring.clone()
PFJet40_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet40/')
PFJet40_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 100.),
)
PFJet40_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet40_v*")

# HLT_PFJet60
PFJet60_Prommonitoring = hltJetMETmonitoring.clone()
PFJet60_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet60/')
PFJet60_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  75 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  150.),
)
PFJet60_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet60_v*")

# HLT_PFJet80
PFJet80_Prommonitoring = hltJetMETmonitoring.clone()
PFJet80_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet80/')
PFJet80_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  200.),
)
PFJet80_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet80_v*")

# HLT_PFJet140
PFJet140_Prommonitoring = hltJetMETmonitoring.clone()
PFJet140_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet140/')
PFJet140_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  70 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  350.),
)
PFJet140_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet140_v*")

# HLT_PFJet200
PFJet200_Prommonitoring = hltJetMETmonitoring.clone()
PFJet200_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet200/')
PFJet200_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  500.),
)
PFJet200_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet200_v*")

# HLT_PFJet260
PFJet260_Prommonitoring = hltJetMETmonitoring.clone()
PFJet260_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet260/')
PFJet260_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  65 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  650.),
)
PFJet260_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet260_v*")

# HLT_PFJet320
PFJet320_Prommonitoring = hltJetMETmonitoring.clone()
PFJet320_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet320/')
PFJet320_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  80 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  800.),
)
PFJet320_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet320_v*")

# HLT_PFJet400
PFJet400_Prommonitoring = hltJetMETmonitoring.clone()
PFJet400_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet400/')
PFJet400_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  1000.),
)
PFJet400_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet400_v*")

# HLT_PFJet500
PFJet500_Prommonitoring = hltJetMETmonitoring.clone()
PFJet500_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_PFJet500/')
PFJet500_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  125),
  xmin  = cms.double(   0.),
  xmax  = cms.double(1250),
)
PFJet500_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet500_v*")

### HLT_PFJetFwd Triggers ###
# HLT_PFJetFwd450
PFJetFwd450_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd450_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd450/')
PFJetFwd450_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  112 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 1120.),
)
PFJetFwd450_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd450_v*")

# HLT_PFJetFwd40
PFJetFwd40_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd40_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd40/')
PFJetFwd40_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 100.),
)
PFJetFwd40_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd40_v*")

# HLT_PFJetFwd60
PFJetFwd60_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd60_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd60/')
PFJetFwd60_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  75 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  150.),
)
PFJetFwd60_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd60_v*")

# HLT_PFJetFwd80
PFJetFwd80_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd80_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd80/')
PFJetFwd80_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  200.),
)
PFJetFwd80_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd80_v*")

# HLT_PFJetFwd140
PFJetFwd140_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd140_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd140/')
PFJetFwd140_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  70 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  350.),
)
PFJetFwd140_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd140_v*")

# HLT_PFJetFwd200
PFJetFwd200_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd200_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd200/')
PFJetFwd200_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  500.),
)
PFJetFwd200_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd200_v*")

# HLT_PFJetFwd260
PFJetFwd260_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd260_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd260/')
PFJetFwd260_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  65 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  650.),
)
PFJetFwd260_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd260_v*")

# HLT_PFJetFwd320
PFJetFwd320_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd320_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd320/')
PFJetFwd320_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  80 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  800.),
)
PFJetFwd320_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd320_v*")

# HLT_PFJetFwd400
PFJetFwd400_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd400_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd400/')
PFJetFwd400_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  1000.),
)
PFJetFwd400_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd400_v*")

# HLT_PFJetFwd500
PFJetFwd500_Prommonitoring = hltJetMETmonitoring.clone()
PFJetFwd500_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd500/')
PFJetFwd500_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  125),
  xmin  = cms.double(   0.),
  xmax  = cms.double(1250),
)
PFJetFwd500_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJetFwd500_v*")

### HLT_AK8 Triggers ###
# HLT_AK8PFJet40
AK8PFJet40_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet40_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet40/')
AK8PFJet40_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet40_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 100.),
)
AK8PFJet40_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet40_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet40_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet40_v*")

# HLT_AK8PFJet60
AK8PFJet60_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet60_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet60/')
AK8PFJet60_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet60_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  75 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  150.),
)
AK8PFJet60_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet60_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet60_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet60_v*")

# HLT_AK8PFJet80
AK8PFJet80_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet80_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet80/')
AK8PFJet80_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet80_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  200.),
)
AK8PFJet80_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet80_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet80_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet80_v*")

# HLT_AK8PFJet140
AK8PFJet140_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet140_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet140/')
AK8PFJet140_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet140_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  70 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  350.),
)
AK8PFJet140_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet140_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet140_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet140_v*")

# HLT_AK8PFJet200
AK8PFJet200_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet200_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet200/')
AK8PFJet200_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet200_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  500.),
)
AK8PFJet200_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet200_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet200_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet200_v*")

# HLT_AK8PFJet260
AK8PFJet260_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet260_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet260/')
AK8PFJet260_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet260_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  65 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  650.),
)
AK8PFJet260_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet260_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet260_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet260_v*")

# HLT_AK8PFJet320
AK8PFJet320_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet320_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet320/')
AK8PFJet320_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet320_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  80 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  800.),
)
AK8PFJet320_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet320_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet320_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet320_v*")

# HLT_AK8PFJet400
AK8PFJet400_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet400_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet400/')
AK8PFJet400_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet400_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  1000.),
)
AK8PFJet400_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet400_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet400_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet400_v*")

# HLT_AK8PFJet450
AK8PFJet450_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet450_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet450/')
AK8PFJet450_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet450_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  112 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 1120.),
)
AK8PFJet450_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet450_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet450_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet450_v*")

# HLT_AK8PFJet500
AK8PFJet500_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet500_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8/PF/HLT_AK8PFJet500/')
AK8PFJet500_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJet500_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  125),
  xmin  = cms.double(   0.),
  xmax  = cms.double(1250),
)
AK8PFJet500_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet500_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet500_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet500_v*")

### HLT_AK8Fwd Triggers ###
# HLT_AK8PFJetFwd40
AK8PFJetFwd40_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd40_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd40/')
AK8PFJetFwd40_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd40_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 100.),
)
AK8PFJetFwd40_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd40_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd40_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd40_v*")

# HLT_AK8PFJetFwd60
AK8PFJetFwd60_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd60_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd60/')
AK8PFJetFwd60_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd60_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  75 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  150.),
)
AK8PFJetFwd60_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd60_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd60_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd60_v*")

# HLT_AK8PFJetFwd80
AK8PFJetFwd80_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd80_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd80/')
AK8PFJetFwd80_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd80_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  200.),
)
AK8PFJetFwd80_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd80_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd80_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd80_v*")

# HLT_AK8PFJetFwd140
AK8PFJetFwd140_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd140_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd140/')
AK8PFJetFwd140_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd140_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  70 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  350.),
)
AK8PFJetFwd140_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd140_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd140_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd140_v*")

# HLT_AK8PFJetFwd200
AK8PFJetFwd200_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd200_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd200/')
AK8PFJetFwd200_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd200_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  500.),
)
AK8PFJetFwd200_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd200_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd200_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd200_v*")

# HLT_AK8PFJetFwd260
AK8PFJetFwd260_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd260_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd260/')
AK8PFJetFwd260_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd260_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  65 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  650.),
)
AK8PFJetFwd260_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd260_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd260_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd260_v*")

# HLT_AK8PFJetFwd320
AK8PFJetFwd320_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd320_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd320/')
AK8PFJetFwd320_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd320_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  80 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  800.),
)
AK8PFJetFwd320_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd320_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd320_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd320_v*")

# HLT_AK8PFJetFwd400
AK8PFJetFwd400_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd400_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd400/')
AK8PFJetFwd400_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd400_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  100 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(  1000.),
)
AK8PFJetFwd400_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd400_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd400_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd400_v*")

# HLT_AK8PFJetFwd450
AK8PFJetFwd450_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd450_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd450/')
AK8PFJetFwd450_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd450_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  112 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double( 1120.),
)
AK8PFJetFwd450_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd450_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd450_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd450_v*")

# HLT_AK8PFJetFwd500
AK8PFJetFwd500_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJetFwd500_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd500/')
AK8PFJetFwd500_Prommonitoring.jetSrc = cms.InputTag("ak8PFJetsCHS")
AK8PFJetFwd500_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  125),
  xmin  = cms.double(   0.),
  xmax  = cms.double(1250),
)
AK8PFJetFwd500_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJetFwd500_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJetFwd500_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJetFwd500_v*")

# HLT_CaloJet500_NoJetID
CaloJet500_NoJetID_Prommonitoring = hltJetMETmonitoring.clone()
CaloJet500_NoJetID_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/Calo/HLT_CaloJet500_NoJetID/')
CaloJet500_NoJetID_Prommonitoring.jetSrc = cms.InputTag("ak4CaloJets")
CaloJet500_NoJetID_Prommonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  125),
  xmin  = cms.double(   0.),
  xmax  = cms.double(1250),
)
CaloJet500_NoJetID_Prommonitoring.ispfjettrg = cms.bool(False)
CaloJet500_NoJetID_Prommonitoring.iscalojettrg = cms.bool(True)
CaloJet500_NoJetID_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_CaloJet500_NoJetID_v*")



HLTJetmonitoring = cms.Sequence(
    PFJet40_Prommonitoring    
    *PFJet60_Prommonitoring    
    *PFJet80_Prommonitoring    
    *PFJet140_Prommonitoring    
    *PFJet200_Prommonitoring    
    *PFJet260_Prommonitoring    
    *PFJet320_Prommonitoring    
    *PFJet400_Prommonitoring
    *PFJet450_Prommonitoring
    *PFJet500_Prommonitoring
    *PFJetFwd40_Prommonitoring    
    *PFJetFwd60_Prommonitoring    
    *PFJetFwd80_Prommonitoring    
    *PFJetFwd140_Prommonitoring    
    *PFJetFwd200_Prommonitoring    
    *PFJetFwd260_Prommonitoring    
    *PFJetFwd320_Prommonitoring    
    *PFJetFwd400_Prommonitoring    
    *PFJetFwd450_Prommonitoring
    *PFJetFwd500_Prommonitoring
    *AK8PFJet450_Prommonitoring
    *AK8PFJet40_Prommonitoring    
    *AK8PFJet60_Prommonitoring    
    *AK8PFJet80_Prommonitoring    
    *AK8PFJet140_Prommonitoring    
    *AK8PFJet200_Prommonitoring    
    *AK8PFJet260_Prommonitoring    
    *AK8PFJet320_Prommonitoring    
    *AK8PFJet400_Prommonitoring    
    *AK8PFJet500_Prommonitoring
    *AK8PFJetFwd450_Prommonitoring
    *AK8PFJetFwd40_Prommonitoring    
    *AK8PFJetFwd60_Prommonitoring    
    *AK8PFJetFwd80_Prommonitoring    
    *AK8PFJetFwd140_Prommonitoring    
    *AK8PFJetFwd200_Prommonitoring    
    *AK8PFJetFwd260_Prommonitoring    
    *AK8PFJetFwd320_Prommonitoring    
    *AK8PFJetFwd400_Prommonitoring    
    *AK8PFJetFwd500_Prommonitoring 
    *CaloJet500_NoJetID_Prommonitoring
)
