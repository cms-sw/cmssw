import FWCore.ParameterSet.Config as cms

#constants for histograms
nTracksBins = 100
nHitsBins = 40
nLayersBins = 20
phiMax = 3.2
phiBinSize = 0.1
etaMax = 2.5
etaBinSize = 0.1
trackPtMax = 50
trackPtBinSize = 0.5
dzMax = 20
dzBinSize = 0.2
dxyMax = 10
dxyBinSize = 0.2
factorMax = 3.0
factorBinSize = 0.1
etMax = 200
etBinSize = 2
eMax = 200
eBinSize = 2
pMax = 200
pBinSize = 2
ptMax = 200
ptBinSize = 2
pcMax = 200
pcBinSize = 2
massMax = 25
massBinSize = 1
deltaEtaMax = 0.5
deltaEtaBinSize = 0.01
deltaPhiMax = 0.5
deltaPhiBinSize = 0.01
sOverNMax = 50
sOverNBinSize = 0.1
drMax = 1.0
jetConeSize = 0.5 #no point having histograms with DR larger than the jet cone size when out of cone tracks are not used
drBinSize = 0.01
ptFractionBins = 50
n90Bins = 50
fHPDBins = 50
resEMFBins = 50
fRBXBins = 50
#derrived constants
nTracksMax = nTracksBins
nHitsMax = nHitsBins
nLayersMax = nLayersBins
factorBins = int((factorMax-1.0)/factorBinSize)
phiMin = -1.*phiMax
phiBins = int(2*phiMax/phiBinSize)
etaMin = -1.*etaMax
etaBins = int(2*etaMax/etaBinSize)
trackPtBins = int(trackPtMax/trackPtBinSize)
dxyMin = -1.*dxyMax
dxyBins = int(2*dxyMax/dxyBinSize)
dzMin = -1.*dzMax
dzBins = int(2*dzMax/dzBinSize)
etBins = int(etMax/etBinSize)
eBins = int(eMax/eBinSize)
pBins = int(pMax/pBinSize)
ptBins = int(ptMax/ptBinSize)
pcBins = int(pcMax/pcBinSize)
massBins = int(massMax/massBinSize)
deltaEtaMin = -1.*deltaEtaMax
deltaEtaBins = int(2*deltaEtaMax/deltaEtaBinSize)
deltaPhiMin = -1.*deltaPhiMax
deltaPhiBins = int(2*deltaPhiMax/deltaPhiBinSize)
sOverNBins = int(sOverNMax/sOverNBinSize)
drBins = int(drMax/drBinSize)
n90Max = n90Bins

import RecoJets.JetProducers.JetIDParams_cfi
theJetIDParams = RecoJets.JetProducers.JetIDParams_cfi.JetIDParams.clone()

#plugin config
jptDQMParameters = cms.PSet(
  #Folder in DQM Store to write histograms to
  HistogramPath = cms.string('JetMET/Jet/uncJPT'),##JPT
  #Whether to dump buffer info and raw data if any error is found
  PrintDebugMessages = cms.untracked.bool(False),
  #Whether to write the DQM store to a file at the end of the run and the file name
  #KH WriteDQMStore = cms.untracked.bool(True), This has to be false by default
  WriteDQMStore = cms.untracked.bool(False),
  DQMStoreFileName = cms.untracked.string('DQMStore.root'),
  
  #Jet ID #here not cleaned
  n90HitsMin = cms.int32(0),
  fHPDMax    = cms.double(1.),
  resEMFMin  = cms.double(0.0),
  correctedPtThreshold = cms.double(3.0),
  JetIDParams = theJetIDParams,
  
  #Historgram configuration
  EHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(eBins),
    Min = cms.double(0),
    Max = cms.double(eMax)
  ),
  EtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etBins),
    Min = cms.double(0),
    Max = cms.double(etMax)
  ),
  PHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(pBins),
    Min = cms.double(0),
    Max = cms.double(pMax)
  ),
  MassHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(massBins),
    Min = cms.double(0),
    Max = cms.double(massMax)
  ),
  PtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  Pt1HistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  Pt2HistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  Pt3HistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptBins),
    Min = cms.double(0),
    Max = cms.double(ptMax)
  ),
  PxHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(pcBins),
    Min = cms.double(0),
    Max = cms.double(pcMax)
  ),
  PyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(pcBins),
    Min = cms.double(0),
    Max = cms.double(pcMax)
  ),
  PzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(pcBins),
    Min = cms.double(0),
    Max = cms.double(pcMax)
  ),
  EtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  PhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax)
  ),
  deltaEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(deltaEtaBins),
    Min = cms.double(deltaEtaMin),
    Max = cms.double(deltaEtaMax)
  ),
  deltaPhiHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(deltaPhiBins),
    Min = cms.double(deltaPhiMin),
    Max = cms.double(deltaPhiMax)
  ),
  PhiVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(phiBins),
    Min = cms.double(phiMin),
    Max = cms.double(phiMax),
    NBinsY = cms.uint32(etaBins),
    MinY = cms.double(etaMin),
    MaxY = cms.double(etaMax)
  ),
  N90HitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(n90Bins),
    Min = cms.double(0),
    Max = cms.double(n90Max)
  ),
  fHPDHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(fHPDBins),
    Min = cms.double(0.0),
    Max = cms.double(1.0)
  ),
  ResEMFHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(resEMFBins),
    Min = cms.double(0.0),
    Max = cms.double(1.0)
  ),
  fRBXHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(fRBXBins),
    Min = cms.double(0.0),
    Max = cms.double(1.0)
  ),
  
  #Pions
  nAllPionsTracksPerJetHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksBins)
  ),
  AllPionsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  AllPionsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  AllPionsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  AllPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  AllPionsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  InCaloInVertexPionsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  InCaloInVertexPionsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  InCaloInVertexPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloInVertexPionsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  InCaloOutVertexPionsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  InCaloOutVertexPionsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  InCaloOutVertexPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloOutVertexPionsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  OutCaloInVertexPionsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  OutCaloInVertexPionsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  OutCaloInVertexPionsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  OutCaloInVertexPionsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
  #  NBins = cms.uint32(nTracksBins),
    NBins = cms.uint32(10),
    Min = cms.double(0),
  #  Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  AllMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  AllMuonsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  AllMuonsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  AllMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  AllMuonsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
  #  NBins = cms.uint32(nTracksBins),
    NBins = cms.uint32(10),
    Min = cms.double(0),
  #  Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  InCaloInVertexMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  InCaloInVertexMuonsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  InCaloInVertexMuonsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  InCaloInVertexMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloInVertexMuonsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
  #  NBins = cms.uint32(nTracksBins),
    NBins = cms.uint32(10),
    Min = cms.double(0),
  #  Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  InCaloOutVertexMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  InCaloOutVertexMuonsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  InCaloOutVertexMuonsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  InCaloOutVertexMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloOutVertexMuonsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
  #  NBins = cms.uint32(nTracksBins),
    NBins = cms.uint32(10),
    Min = cms.double(0),
  #  Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  OutCaloInVertexMuonsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  OutCaloInVertexMuonsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  OutCaloInVertexMuonsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  OutCaloInVertexMuonsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  OutCaloInVertexMuonsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
   # NBins = cms.uint32(nTracksBins), 
    NBins = cms.uint32(10),
    Min = cms.double(0),
   # Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  AllElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  AllElectronsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  AllElectronsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  AllElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  AllElectronsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
  #  NBins = cms.uint32(nTracksBins),
    NBins = cms.uint32(10),
    Min = cms.double(0),
  #  Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  InCaloInVertexElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  InCaloInVertexElectronsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  InCaloInVertexElectronsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  InCaloInVertexElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloInVertexElectronsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
  #  NBins = cms.uint32(nTracksBins),
    NBins = cms.uint32(10),
    Min = cms.double(0),
  #  Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  InCaloOutVertexElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  InCaloOutVertexElectronsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  InCaloOutVertexElectronsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  InCaloOutVertexElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  InCaloOutVertexElectronsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
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
  #  NBins = cms.uint32(nTracksBins),
    NBins = cms.uint32(10),
    Min = cms.double(0),
  #  Max = cms.double(nTracksBins)
    Max = cms.double(10)
  ),
  OutCaloInVertexElectronsTrackPtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(trackPtBins),
    Min = cms.double(0),
    Max = cms.double(trackPtMax)
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
  OutCaloInVertexElectronsTrackDzHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dzBins),
    Min = cms.double(dzMin),
    Max = cms.double(dzMax)
  ),
  OutCaloInVertexElectronsTrackDxyHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(dxyBins),
    Min = cms.double(dxyMin),
    Max = cms.double(dxyMax)
  ),
  OutCaloInVertexElectronsTrackNHitsHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nHitsBins),
    Min = cms.double(0), 
    Max = cms.double(nHitsMax)
  ),
  OutCaloInVertexElectronsTrackNLayersHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nLayersBins),
    Min = cms.double(0), 
    Max = cms.double(nLayersMax)
  ),
  OutCaloInVertexElectronsTrackPtVsEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  
  #Jet level histograms
  nTracksHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(nTracksBins),
    Min = cms.double(0),
    Max = cms.double(nTracksMax)
  ),
  nTracksVsJetEtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etBins),
    Min = cms.double(0),
    Max = cms.double(etMax)
  ),
  nTracksVsJetEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  PtFractionInConeHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(ptFractionBins),
    Min = cms.double(0),
    Max = cms.double(1.0)
  ),
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
  CorrFactorHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(factorBins),
    Min = cms.double(1.0),
    Max = cms.double(factorMax)
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
  ZSPCorrFactorHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(factorBins),
    Min = cms.double(1.0),
    Max = cms.double(factorMax)
  ),
  ZSPCorrFactorVsJetEtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etBins),
    Min = cms.double(0),
    Max = cms.double(etMax)
  ),
  ZSPCorrFactorVsJetEtaHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etaBins),
    Min = cms.double(etaMin),
    Max = cms.double(etaMax)
  ),
  JPTCorrFactorHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(factorBins),
    Min = cms.double(1.0),
    Max = cms.double(factorMax)
  ),
  JPTCorrFactorVsJetEtHistogramConfig = cms.PSet(
    Enabled = cms.bool(True),
    NBins = cms.uint32(etBins),
    Min = cms.double(0),
    Max = cms.double(etMax)
  ),
  JPTCorrFactorVsJetEtaHistogramConfig = cms.PSet(
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

cleanedjptDQMParameters = jptDQMParameters.clone(
  HistogramPath = cms.string('JetMET/Jet/JPT'),
  n90HitsMin = cms.int32(2),
  fHPDMax    = cms.double(0.98),
  resEMFMin  = cms.double(0.01),
  correctedPtThreshold = cms.double(3.0)
    )
