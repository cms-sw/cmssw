import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DiJetMonitor_cfi import DiPFjetAve40_Prommonitoring
### HLT_DiJet Triggers ###
# DiPFjetAve60
DiPFjetAve60_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve60/'
)
DiPFjetAve60_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 75,
  xmin  =  0.,
  xmax  = 150.,
)
DiPFjetAve60_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve60_v*")

# DiPFjetAve80
DiPFjetAve80_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve80/'
)
DiPFjetAve80_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 100 ,
  xmin  = 0.,
  xmax  =  200.,
)
DiPFjetAve80_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve80_v*")

# DiPFjetAve140
DiPFjetAve140_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve140/'
)
DiPFjetAve140_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins =  70 ,
  xmin  =  0.,
  xmax  =  350.,
)
DiPFjetAve140_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve140_v*")

# DiPFjetAve200
DiPFjetAve200_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve200/'
)
DiPFjetAve200_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 50 ,
  xmin  = 0.,
  xmax  =  500.,
)
DiPFjetAve200_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve200_v*")

# DiPFjetAve260
DiPFjetAve260_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve260/'
)
DiPFjetAve260_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 65 ,
  xmin  =  0.,
  xmax  = 650.,
)
DiPFjetAve260_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve260_v*")

# DiPFjetAve320
DiPFjetAve320_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve320/'
)
DiPFjetAve320_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 80 ,
  xmin  =  0.,
  xmax  =  800.,
)
DiPFjetAve320_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve320_v*")

# DiPFjetAve400
DiPFjetAve400_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve400/'
)
DiPFjetAve400_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 100 ,
  xmin  =  0.,
  xmax  = 1000.,
)
DiPFjetAve400_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve400_v*")

# DiPFjetAve500
DiPFjetAve500_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve500/'
)
DiPFjetAve500_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 125,
  xmin  =  0.,
  xmax  = 1250,
)
DiPFjetAve500_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve500_v*")

# HLT_DiPFJetAve60_HFJEC
DiPFjetAve60_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve60_HFJEC/'
)
DiPFjetAve60_HFJEC_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 75 ,
  xmin  =  0.,
  xmax  =  150.,
)
DiPFjetAve60_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve60_HFJEC_v*")

# HLT_DiPFJetAve80_HFJEC
DiPFjetAve80_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve80_HFJEC/'
)
DiPFjetAve80_HFJEC_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 100 ,
  xmin  =  0.,
  xmax  =  200.,
)
DiPFjetAve80_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve80_HFJEC_v*")

# HLT_DiPFJetAve100_HFJEC
DiPFjetAve100_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve100_HFJEC/'
)
DiPFjetAve100_HFJEC_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 50 ,
  xmin  =  0.,
  xmax  =  250.,
)
DiPFjetAve100_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve100_HFJEC_v*")

# HLT_DiPFJetAve160_HFJEC
DiPFjetAve160_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve160_HFJEC/'
)
DiPFjetAve160_HFJEC_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 80 ,
  xmin  =  0.,
  xmax  =  400.,
)
DiPFjetAve160_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve160_HFJEC_v*")

# HLT_DiPFJetAve220_HFJEC
DiPFjetAve220_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve220_HFJEC/'
)
DiPFjetAve220_HFJEC_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 55 ,
  xmin  =  0.,
  xmax  =  550.,
)
DiPFjetAve220_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve220_HFJEC_v*")

# HLT_DiPFJetAve300_HFJEC
DiPFjetAve300_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve300_HFJEC/'
)
DiPFjetAve300_HFJEC_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 75 ,
  xmin  =  0.,
  xmax  = 750.,
)
DiPFjetAve300_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve300_HFJEC_v*")

HLTDiJetmonitoring = cms.Sequence(
    DiPFjetAve40_Prommonitoring
    *DiPFjetAve60_Prommonitoring
    *DiPFjetAve80_Prommonitoring
    *DiPFjetAve140_Prommonitoring
    *DiPFjetAve200_Prommonitoring
    *DiPFjetAve260_Prommonitoring
    *DiPFjetAve320_Prommonitoring
    *DiPFjetAve400_Prommonitoring
    *DiPFjetAve500_Prommonitoring
    *DiPFjetAve60_HFJEC_Prommonitoring
    *DiPFjetAve80_HFJEC_Prommonitoring
    *DiPFjetAve100_HFJEC_Prommonitoring
    *DiPFjetAve160_HFJEC_Prommonitoring
    *DiPFjetAve220_HFJEC_Prommonitoring
    *DiPFjetAve300_HFJEC_Prommonitoring
)

