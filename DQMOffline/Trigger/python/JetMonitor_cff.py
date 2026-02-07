import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring
from JetMETCorrections.Configuration.JetCorrectors_cff import *
from DQMOffline.Trigger.ZGammaplusJetsMonitor_cff import *

### HLT_PFJet Triggers ###
# HLT_PFJet450
PFJet450_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet450/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  112 ,
                xmin  =   0.,
                xmax  = 1120.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet450_v*"])
)


# HLT_PFJet40
PFJet40_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet40/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  50 ,
                xmin  =  0.,
                xmax  = 100.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet40_v*"])
)

# HLT_PFJet60
PFJet60_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet60/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  75 ,
                xmin  =  0.,
                xmax  =  150.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet60_v*"])
)


# HLT_PFJet80
PFJet80_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet80/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  100,
                xmin  =  0.,
                xmax  =  200.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet80_v*"])
)

# HLT_PFJet140
PFJet140_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet140/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  70 ,
                xmin  =  0.,
                xmax  =  350.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet140_v*"])
)

# HLT_PFJet200
PFJet200_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet200/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  50 ,
                xmin  =   0.,
                xmax  =  500.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet200_v*"])
)


# HLT_PFJet260
PFJet260_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet260/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 65,
                xmin  = 0.,
                xmax  =  650.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet260_v*"])

)

# HLT_PFJet320
PFJet320_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet320/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 80 ,
                xmin  =  0.,
                xmax  = 800.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet320_v*"])
)


# HLT_PFJet400
PFJet400_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet400/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 100 ,
                xmin  =  0.,
                xmax  =  1000.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet400_v*"])
)

# HLT_PFJet500
PFJet500_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_PFJet500/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  125,
                xmin  =  0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet500_v*"])
)


### HLT_PFJetFwd Triggers ###
# HLT_PFJetFwd450
PFJetFwd450_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd450/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  112 ,
                xmin  =  0.,
                xmax  = 1120.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd450_v*"])
)

# HLT_PFJetFwd40
PFJetFwd40_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd40/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  50 ,
                xmin  =   0.,
                xmax  = 100.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd40_v*"])
)


# HLT_PFJetFwd60
PFJetFwd60_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd60/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  75 ,
                xmin  =   0.,
                xmax  =  150.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd60_v*"])
)


# HLT_PFJetFwd80
PFJetFwd80_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd80/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  100,
                xmin  =   0.,
                xmax  =  200.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd80_v*"])

)

# HLT_PFJetFwd140
PFJetFwd140_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd140/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  70 ,
                xmin  =   0.,
                xmax  =  350.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd140_v*"])
)


# HLT_PFJetFwd200
PFJetFwd200_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd200/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 50 ,
                xmin  =   0.,
                xmax  =  500.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd200_v*"])
)


# HLT_PFJetFwd260
PFJetFwd260_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd260/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 65 ,
                xmin  =  0.,
                xmax  = 650.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd260_v*"])
)


# HLT_PFJetFwd320
PFJetFwd320_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd320/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 80 ,
                xmin  =  0.,
                xmax  = 800.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd320_v*"])
)


# HLT_PFJetFwd400
PFJetFwd400_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd400/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 100 ,
                xmin  =  0.,
                xmax  =  1000.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd400_v*"])
)


# HLT_PFJetFwd500
PFJetFwd500_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4Fwd/PF/HLT_PFJetFwd500/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 125,
                xmin  =   0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJetFwd500_v*"])
)


### HLT_AK8 Triggers ###
# HLT_AK8PFJet40
AK8PFJet40_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet40/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet =dict(jetPtThrPSet = dict(
                nbins =  50 ,
                xmin  =   0.,
                xmax  = 100.)),

    ispfjettrg = True,
    iscalojettrg = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet40_v*"])
)


# HLT_AK8PFJet60
AK8PFJet60_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet60/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,  # flag for the trigger path 
    iscalojettrg = False,
    ispuppijet = True,  # flag for offline collection and histonaming
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  75 ,
                xmin  =  0.,
                xmax  =  150.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet60_v*"])
)


# HLT_AK8PFJet80
AK8PFJet80_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet80/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  100 ,
                xmin  =   0.,
                xmax  =  200.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet80_v*"])

)

# HLT_AK8PFJet140
AK8PFJet140_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet140/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  70 ,
                xmin  =   0.,
                xmax  = 350.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet140_v*"])
)


# HLT_AK8PFJet200
AK8PFJet200_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet200/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  50,
                xmin  =  0.,
                xmax  =  500.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet200_v*"])
)


# HLT_AK8PFJet260
AK8PFJet260_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet260/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  65 ,
                xmin  =   0.,
                xmax  =  650.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet260_v*"])
)


# HLT_AK8PFJet320
AK8PFJet320_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet320/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  80 ,
                xmin  =  0.,
                xmax  = 800.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet320_v*"])
)


# HLT_AK8PFJet400
AK8PFJet400_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet400/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  100 ,
                xmin  =  0.,
                xmax  =  1000.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet400_v*"])
)


# HLT_AK8PFJet450
AK8PFJet450_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet450/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  112 ,
                xmin  =   0.,
                xmax  =  1120.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet450_v*"])
)


# HLT_AK8PFJet500
AK8PFJet500_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8/PF/HLT_AK8PFJet500/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  125,
                xmin  =  0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet500_v*"])

)

### HLT_AK8Fwd Triggers ###
# HLT_AK8PFJetFwd40
AK8PFJetFwd40_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd40/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  50 ,
                xmin  =  0.,
                xmax  = 100.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd40_v*"])
)


# HLT_AK8PFJetFwd60
AK8PFJetFwd60_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd60/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  75 ,
                xmin  =   0.,
                xmax  =  150.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd60_v*"])
)


# HLT_AK8PFJetFwd80
AK8PFJetFwd80_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd80/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  100 ,
                xmin  =   0.,
                xmax  =  200.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd80_v*"])
)


# HLT_AK8PFJetFwd140
AK8PFJetFwd140_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd140/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  70 ,
                xmin  =   0.,
                xmax  =  350.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd140_v*"])
)


# HLT_AK8PFJetFwd200
AK8PFJetFwd200_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd200/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  50 ,
                xmin  =   0.,
                xmax  =  500.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd200_v*"])
)


# HLT_AK8PFJetFwd260
AK8PFJetFwd260_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd260/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  65 ,
                xmin  =   0.,
                xmax  =  650.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd260_v*"])
)


# HLT_AK8PFJetFwd320
AK8PFJetFwd320_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd320/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  80 ,
                xmin  =  0.,
                xmax  = 800.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd320_v*"])
)


# HLT_AK8PFJetFwd400
AK8PFJetFwd400_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd400/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  100 ,
                xmin  =   0.,
                xmax  =  1000.)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd400_v*"])
)

# HLT_AK8PFJetFwd450
AK8PFJetFwd450_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd450/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =   112 ,
                xmin  =  0.,
                xmax  = 1120.)),
    numGenericTriggerEventPSet = dict(hltPaths =["HLT_AK8PFJetFwd450_v*"])
)


# HLT_AK8PFJetFwd500
AK8PFJetFwd500_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK8Fwd/PF/HLT_AK8PFJetFwd500/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispfjettrg = True,
    iscalojettrg = False,
    ispuppijet = True,
    enableFullMonitoring = False,
    dr2cut = 0.64,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  125,
                xmin  =   0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJetFwd500_v*"])

)

# HLT_CaloJet500_NoJetID
CaloJet500_NoJetID_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/Calo/HLT_CaloJet500_NoJetID/',
    jetSrc = "ak4CaloJets",
    corrector = "ak4CaloL1FastL2L3ResidualCorrector",
    ispfjettrg = False,
    iscalojettrg = True,
    ispuppijet = False,
    enableFullMonitoring = False,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  125,
                xmin  =   0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloJet500_NoJetID_v*"])
)
# configure 3 unprescaled triggers to monitor them using orthogonal method with muon dataset
# ----------------------------------------------------------------------------------------
# HLT_PFJetFwd500
PFJet500_Orthogonal_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/MuonOrthogonal/HLT_PFJet500/',
    corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = True,
    dr2cut = 0.16,
    nmuons = 1,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins = 125,
                xmin  =   0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFJet500_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_IsoMu24_v*"])
)
# HLT_AK8PFJet500
AK8PFJet500_Orthogonal_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/MuonOrthogonal/HLT_AK8PFJet500/',
    jetSrc = "ak8PFJetsPuppi",
    corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
    ispuppijet = True,
    enableFullMonitoring = False,
    ispfjettrg = True,
    iscalojettrg = False,
    dr2cut = 0.64,
    nmuons = 1,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  125,
                xmin  =  0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet500_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_IsoMu24_v*"])
)
# HLT_CaloJet500_NoJetID
CaloJet500_NoJetID_Orthogonal_Prommonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/JME/Jets/MuonOrthogonal/HLT_CaloJet500_NoJetID/',
    jetSrc = "ak4CaloJets",
    corrector = "ak4CaloL1FastL2L3ResidualCorrector",
    ispuppijet = False,
    enableFullMonitoring = True,
    ispfjettrg = False,
    iscalojettrg = True,
    dr2cut = 0.16,
    nmuons = 1,
    histoPSet = dict(jetPtThrPSet = dict(
                nbins =  125,
                xmin  =   0.,
                xmax  = 1250)),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloJet500_NoJetID_v*"]),
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_IsoMu24_v*"])
)


HLTJetmonitoring = cms.Sequence(    
    ak4PFPuppiL1FastL2L3ResidualCorrectorChain
    *PFJet500_Orthogonal_Prommonitoring
    *PFJet40_Prommonitoring    
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
    *ak8PFPuppiL1FastL2L3ResidualCorrectorChain
    *AK8PFJet500_Orthogonal_Prommonitoring
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
    *ak4CaloL1FastL2L3ResidualCorrectorChain
    *CaloJet500_NoJetID_Prommonitoring
    *CaloJet500_NoJetID_Orthogonal_Prommonitoring
)
