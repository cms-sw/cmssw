import FWCore.ParameterSet.Config as cms
from DQMOffline.Trigger.SoftOSMonitor_cfi import hltSOSmonitoring

double_soft_muon_muonpt = hltSOSmonitoring.clone()
double_soft_muon_muonpt.FolderName   = cms.string('HLT/SUSY/SOS/Muon/')
# Selections
double_soft_muon_muonpt.nmuons           = cms.int32(2)
double_soft_muon_muonpt.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_muonpt.HTcut            = cms.double(60)
double_soft_muon_muonpt.metSelection     =cms.string('pt>150')
double_soft_muon_muonpt.MHTdefinition    = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_muonpt.MHTcut           = cms.double(150)
double_soft_muon_muonpt.invMassUppercut  = cms.double(50)
double_soft_muon_muonpt.invMassLowercut  = cms.double(10)
double_soft_muon_muonpt.met_pt_cut=cms.int32(160)
double_soft_muon_muonpt.turn_on=cms.string("mu")

double_soft_muon_muonpt.histoPSet.sosPSet=cms.PSet(nbins=cms.int32(20), xmin=cms.double(0.0), xmax=cms.double(60) )

double_soft_muon_muonpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_muonpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_v*')

#met
double_soft_muon_metpt = hltSOSmonitoring.clone()
double_soft_muon_metpt.FolderName   = cms.string('HLT/SUSY/SOS/MET/')
# Selections
double_soft_muon_metpt.nmuons           = cms.int32(2)
double_soft_muon_metpt.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_metpt.HTcut            = cms.double(60)
double_soft_muon_metpt.muoSelection     =cms.string('pt>18 & abs(eta)<2.4')
double_soft_muon_metpt.MHTdefinition    = cms.string('pt>30 & abs(eta)<2.4')
double_soft_muon_metpt.MHTcut           = cms.double(150)
double_soft_muon_metpt.invMassUppercut       = cms.double(50)
double_soft_muon_metpt.invMassLowercut       = cms.double(10)
# Binning
double_soft_muon_metpt.histoPSet.sosPSet   =cms.PSet(nbins=cms.int32(50),xmin=cms.double(50),xmax=cms.double(300) )
# Triggers
double_soft_muon_metpt.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v*')
double_soft_muon_metpt.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*')


from DQMOffline.Trigger.RazorMonitor_cff import *

susyMonitorHLT = cms.Sequence(

    susyHLTRazorMonitoring
+double_soft_muon_metpt
+double_soft_muon_muonpt

)
