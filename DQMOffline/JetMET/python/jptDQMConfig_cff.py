import FWCore.ParameterSet.Config as cms

#constants for histograms
nTracksBins = 100
nHitsBins = 40
phiMax = 3.2
phiBinSize = 0.1
etaMax = 2.5
etaBinSize = 0.1
ptMax = 50
ptBinSize = 0.1
etMax = 200
etBinSize = 1
sOverNMax = 50
sOverNBinSize = 0.1
drMax = 1.0
jetConeSize = 0.5 #no point having histograms with DR larger than the jet cone size when out of cone tracks are not used
drBinSize = 0.01
#derrived constants
nTracksMax = nTracksBins
nHitsMax = nHitsBins
phiMin = -phiMax
phiBins = int(2*phiMax/phiBinSize)
etaMin = -etaMax
etaBins = int(2*etaMax/etaBinSize)
ptBins = int(ptMax/ptBinSize)
etBins = int(etMax/etBinSize)
sOverNBins = int(sOverNMax/sOverNBinSize)
drBins = int(drMax/drBinSize)

import JetMETCorrections.Configuration.JetPlusTrackCorrections_cff
JetPlusTrackZSPCorrectorIcone5ForDQM = JetMETCorrections.Configuration.JetPlusTrackCorrections_cff.JetPlusTrackZSPCorrectorIcone5.clone()
JetPlusTrackZSPCorrectorIcone5ForDQM.ElectronIds = 'eidTight'
JetPlusTrackZSPCorrectorIcone5ForDQM.label = 'JetPlusTrackZSPCorrectorIcone5ForDQM'
JetPlusTrackZSPCorrectorIcone5ForDQM.JetTracksAssociationAtVertex = cms.InputTag('iterativeCone5JetTracksAssociatorAtVertex')
JetPlusTrackZSPCorrectorIcone5ForDQM.JetTracksAssociationAtCaloFace = cms.InputTag('iterativeCone5JetTracksAssociatorAtCaloFace')
from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *
from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *

#plugin config
jptDQMParameters = cms.PSet(
  #Folder in DQM Store to write histograms to
  HistogramPath = cms.string('JetMET/Jet/JPT/'),
  #Whether to dump buffer info and raw data if any error is found
  PrintDebugMessages = cms.untracked.bool(False),
  #JPT corrector
  JPTCorrectorName = cms.string('JetPlusTrackZSPCorrectorIcone5ForDQM'),
  #ZSP corrector
  ZSPCorrectorName = cms.string('ZSPJetCorrectorIcone5'),
  #Whether to write the DQM store to a file at the end of the run and the file name
  WriteDQMStore = cms.untracked.bool(True),
  DQMStoreFileName = cms.untracked.string('DQMStore.root'),

  #Historgram configuration
  
  #Pions
  #InVertexPionTrackImpactPointJetDRHistogramConfig = cms.PSet(
  #  Enabled = cms.bool(True),
  #  NBins = cms.uint32(drBins),
  #  Min = cms.double(0),
  #  Max = cms.double(drMax)
  #),
  #OutVertexPionTrackImpactPointJetDRHistogramConfig = cms.PSet(
  #  Enabled = cms.bool(True),
  #  NBins = cms.uint32(drBins),
  #  Min = cms.double(0),
  #  Max = cms.double(drMax)
  #),
  nAllPionsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  AllPionsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  AllPionsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  AllPionsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  AllPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  AllPionsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Pions In of cone at calo, in cone at vertex
  nInCaloInVertexPionsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  InCaloInVertexPionsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  InCaloInVertexPionsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  InCaloInVertexPionsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  InCaloInVertexPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloInVertexPionsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Pions In of cone at calo, out cone at vertex
  nInCaloOutVertexPionsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  InCaloOutVertexPionsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  InCaloOutVertexPionsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  InCaloOutVertexPionsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  InCaloOutVertexPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloOutVertexPionsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Pions Out of cone at calo, in cone at vertex
  nOutCaloInVertexPionsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  OutCaloInVertexPionsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  OutCaloInVertexPionsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  OutCaloInVertexPionsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  OutCaloInVertexPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  OutCaloInVertexPionsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  
  #Muons
  nAllMuonsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  AllMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  AllMuonsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  AllMuonsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  AllMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  AllMuonsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Muons In of cone at calo, in cone at vertex
  nInCaloInVertexMuonsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  InCaloInVertexMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  InCaloInVertexMuonsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  InCaloInVertexMuonsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  InCaloInVertexMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloInVertexMuonsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Muons In of cone at calo, out cone at vertex
  nInCaloOutVertexMuonsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  InCaloOutVertexMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  InCaloOutVertexMuonsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  InCaloOutVertexMuonsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  InCaloOutVertexMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloOutVertexMuonsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Muons Out of cone at calo, in cone at vertex
  nOutCaloInVertexMuonsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  OutCaloInVertexMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  OutCaloInVertexMuonsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  OutCaloInVertexMuonsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  OutCaloInVertexMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  OutCaloInVertexMuonsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  
  #Electrons
  nAllElectronsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  AllElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  AllElectronsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  AllElectronsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  AllElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  AllElectronsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Electrons In of cone at calo, in cone at vertex
  nInCaloInVertexElectronsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  InCaloInVertexElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  InCaloInVertexElectronsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  InCaloInVertexElectronsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  InCaloInVertexElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloInVertexElectronsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Electrons In of cone at calo, out cone at vertex
  nInCaloOutVertexElectronsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  InCaloOutVertexElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  InCaloOutVertexElectronsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  InCaloOutVertexElectronsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  InCaloOutVertexElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloOutVertexElectronsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  #Electrons Out of cone at calo, in cone at vertex
  nOutCaloInVertexElectronsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  OutCaloInVertexElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  OutCaloInVertexElectronsTrackPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  OutCaloInVertexElectronsTrackEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  OutCaloInVertexElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  OutCaloInVertexElectronsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  
  #Jet level histograms
  PtFractionInConeVsJetRawEtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etBins),
    Min = cms.double(0),
    Max = cms.double(etMax)
  ),
  PtFractionInConeVsJetEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  CorrFactorVsJetEtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etBins),
    Min = cms.double(0),
    Max = cms.double(etMax)
  ),
  CorrFactorVsJetEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  TrackSiStripHitStoNHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(sOverNBins),
    Min = cms.double(0),
    Max = cms.double(sOverNMax)
  ),
  InCaloTrackDirectionJetDRHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(drBins),
    Min = cms.double(0),
    Max = cms.double(drMax)
  ),
  OutCaloTrackDirectionJetDRHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(drBins),
    Min = cms.double(0),
    Max = cms.double(jetConeSize)
  ), 
  InVertexTrackImpactPointJetDRHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(drBins),
    Min = cms.double(0),
    Max = cms.double(drMax)
  ),
  OutVertexTrackImpactPointJetDRHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(drBins),
    Min = cms.double(0),
    Max = cms.double(jetConeSize)
  ),
)
