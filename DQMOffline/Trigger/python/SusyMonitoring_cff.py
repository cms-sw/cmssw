import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.RazorMonitor_cff import *


from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring


#george
#muon
double_soft_muon_muonpt = hltTOPmonitoring.clone()
double_soft_muon_muonpt.FolderName   = cms.string('HLT/SUSY/SOS/Muon/')
# Selections
double_soft_muon_muonpt.nmuons           = cms.uint32(2)
double_soft_muon_muonpt.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_muonpt.HTcut            = cms.double(60)
double_soft_muon_muonpt.metSelection     =cms.string('pt>150')
double_soft_muon_muonpt.MHTdefinition    = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_muonpt.MHTcut           = cms.double(150)
double_soft_muon_muonpt.invMassUppercut  = cms.double(50)
double_soft_muon_muonpt.invMassLowercut  = cms.double(10)
# Binning
double_soft_muon_muonpt.histoPSet.muPtBinning      =cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)
double_soft_muon_muonpt.histoPSet.muPtBinning2D    =cms.vdouble(0,2,5,7,10,12,15,17,20,25,30,50)
# Triggers
double_soft_muon_muonpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_muonpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_v*')

#met
double_soft_muon_metpt = hltTOPmonitoring.clone()
double_soft_muon_metpt.FolderName   = cms.string('HLT/SUSY/SOS/MET/')
# Selections
double_soft_muon_metpt.nmuons           = cms.uint32(2)
double_soft_muon_metpt.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_metpt.HTcut            = cms.double(60)
double_soft_muon_metpt.muoSelection     =cms.string('pt>18 & abs(eta)<2.4')
double_soft_muon_metpt.MHTdefinition    = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_metpt.MHTcut           = cms.double(150)
double_soft_muon_metpt.invMassUppercut       = cms.double(50)
double_soft_muon_metpt.invMassLowercut       = cms.double(10)
# Binning
double_soft_muon_metpt.histoPSet.metPSet   =cms.PSet(nbins=cms.uint32(50),xmin=cms.double(50),xmax=cms.double(300) )
# Triggers
double_soft_muon_metpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_metpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')

#inv Mass
double_soft_muon_mll = hltTOPmonitoring.clone()
double_soft_muon_mll.FolderName   = cms.string('HLT/SUSY/SOS/Mll/')
# Selections
double_soft_muon_mll.nmuons           = cms.uint32(2)
double_soft_muon_mll.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_mll.HTcut            = cms.double(60)
double_soft_muon_mll.muoSelection     =cms.string('pt>10 & abs(eta)<2.4')
double_soft_muon_mll.MHTdefinition    = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_mll.MHTcut            = cms.double(150)
double_soft_muon_mll.metSelection      = cms.string('pt>150')
# Binning

# Triggers
double_soft_muon_mll.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_mll.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_TripleMu_10_5_5_DZ_v*')

#mht
double_soft_muon_mhtpt = hltTOPmonitoring.clone()
double_soft_muon_mhtpt.FolderName   = cms.string('HLT/SUSY/SOS/MHT/')
# Selections
double_soft_muon_mhtpt.nmuons           = cms.uint32(2)
double_soft_muon_mhtpt.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_mhtpt.HTcut            = cms.double(60)
double_soft_muon_mhtpt.muoSelection     =cms.string('pt>18 & abs(eta)<2.0')
double_soft_muon_mhtpt.MHTdefinition    = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_mhtpt.metSelection     = cms.string('pt>150')
double_soft_muon_mhtpt.invMassUppercut       = cms.double(50)
double_soft_muon_mhtpt.invMassLowercut       = cms.double(10)
# Binning

# Triggers
double_soft_muon_mhtpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_mhtpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')



susyMonitorHLT = cms.Sequence(
    susyHLTRazorMonitoring
+double_soft_muon_muonpt
+double_soft_muon_metpt
+double_soft_muon_mhtpt
+double_soft_muon_mll
)
