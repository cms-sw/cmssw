import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltTOPmonitoring = topMonitoring.clone()

hltTOPmonitoring.FolderName = 'HLT/TOP/default/'
hltTOPmonitoring.requireValidHLTPaths = True

# histo PSets
hltTOPmonitoring.histoPSet.lsPSet.nbins =  250
hltTOPmonitoring.histoPSet.lsPSet.xmin  =    0
hltTOPmonitoring.histoPSet.lsPSet.xmax  = 2500

hltTOPmonitoring.histoPSet.metPSet.nbins =  30
hltTOPmonitoring.histoPSet.metPSet.xmin  =   0
hltTOPmonitoring.histoPSet.metPSet.xmax  = 300

hltTOPmonitoring.histoPSet.ptPSet.nbins =  60
hltTOPmonitoring.histoPSet.ptPSet.xmin  =   0
hltTOPmonitoring.histoPSet.ptPSet.xmax  = 300

hltTOPmonitoring.histoPSet.lsPSet.nbins = 2500

hltTOPmonitoring.histoPSet.phiPSet.nbins = 32
hltTOPmonitoring.histoPSet.phiPSet.xmin  = -3.2
hltTOPmonitoring.histoPSet.phiPSet.xmax  =  3.2

hltTOPmonitoring.histoPSet.etaPSet.nbins = 24
hltTOPmonitoring.histoPSet.etaPSet.xmin  = -2.4
hltTOPmonitoring.histoPSet.etaPSet.xmax  =  2.4

hltTOPmonitoring.histoPSet.htPSet.nbins =  60
hltTOPmonitoring.histoPSet.htPSet.xmin  =   0
hltTOPmonitoring.histoPSet.htPSet.xmax  = 600

hltTOPmonitoring.histoPSet.csvPSet.nbins = 50
hltTOPmonitoring.histoPSet.csvPSet.xmin  =  0
hltTOPmonitoring.histoPSet.csvPSet.xmax  =  1

hltTOPmonitoring.histoPSet.DRPSet.nbins = 60
hltTOPmonitoring.histoPSet.DRPSet.xmin  =  0
hltTOPmonitoring.histoPSet.DRPSet.xmax  =  6

hltTOPmonitoring.histoPSet.invMassPSet.nbins = 40
hltTOPmonitoring.histoPSet.invMassPSet.xmin  =  0
hltTOPmonitoring.histoPSet.invMassPSet.xmax  = 80

hltTOPmonitoring.histoPSet.MHTPSet.nbins =  80
hltTOPmonitoring.histoPSet.MHTPSet.xmin  =  60
hltTOPmonitoring.histoPSet.MHTPSet.xmax  = 300

# MET and HT binning
hltTOPmonitoring.histoPSet.metBinning = [0,20,40,60,80,100,125,150,175,200]
hltTOPmonitoring.histoPSet.HTBinning  = [0,20,40,60,80,100,125,150,175,200,300,400,500,700]
# Eta binning
hltTOPmonitoring.histoPSet.eleEtaBinning = [-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4]
hltTOPmonitoring.histoPSet.jetEtaBinning = [-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4]
hltTOPmonitoring.histoPSet.muEtaBinning  = [-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4]
# pt binning
hltTOPmonitoring.histoPSet.elePtBinning = [0,5,10,20,30,40,50,70,100,200,400]
hltTOPmonitoring.histoPSet.jetPtBinning = [0,5,10,20,30,40,50,70,100,200,400]
hltTOPmonitoring.histoPSet.muPtBinning  = [0,5,10,20,30,40,50,70,100,200,400]
# Eta binning 2D
hltTOPmonitoring.histoPSet.eleEtaBinning2D = [-2.5,-1.5,-0.6,0.,0.6,1.5,2.5]
hltTOPmonitoring.histoPSet.jetEtaBinning2D = [-2.5,-1.5,-0.6,0.,0.6,1.5,2.5]
hltTOPmonitoring.histoPSet.muEtaBinning2D  = [-2.5,-1.5,-0.6,0.,0.6,1.5,2.5]
hltTOPmonitoring.histoPSet.phoEtaBinning2D = [-2.5,-1.5,-0.6,0.,0.6,1.5,2.5]
# pt binning 2D
hltTOPmonitoring.histoPSet.elePtBinning2D = [0,20,30,50,100,200,400]
hltTOPmonitoring.histoPSet.jetPtBinning2D = [0,20,30,50,100,200,400]
hltTOPmonitoring.histoPSet.muPtBinning2D  = [0,20,30,50,100,200,400]
hltTOPmonitoring.histoPSet.phoPtBinning2D = [0,20,30,50,100,200,400]
# HT and phi binning 2D
hltTOPmonitoring.histoPSet.HTBinning2D  = [0,20,40,70,100,150,200,400,700]
hltTOPmonitoring.histoPSet.phiBinning2D = [-3.1416,-1.8849,-0.6283,0.6283,1.8849,3.1416]

hltTOPmonitoring.enablePhotonPlot = False
hltTOPmonitoring.enableMETPlot = False

hltTOPmonitoring.applyLeptonPVcuts = False
hltTOPmonitoring.leptonPVcuts.dxy = 9999.
hltTOPmonitoring.leptonPVcuts.dz  = 9999.

hltTOPmonitoring.met       = "pfMetEI" # pfMet
hltTOPmonitoring.jets      = "ak4PFJetsCHS" # ak4PFJets, ak4PFJetsCHS, pfJetsEI
hltTOPmonitoring.electrons = "gedGsfElectrons" # while pfIsolatedElectronsEI are reco::PFCandidate !
hltTOPmonitoring.elecID    = "egmGsfElectronIDsForDQM:cutBasedElectronID-Fall17-94X-V1-tight" #Electron ID
hltTOPmonitoring.muons     = "muons" # while pfIsolatedMuonsEI are reco::PFCandidate !
hltTOPmonitoring.photons   = "photons" # reco::Photon
hltTOPmonitoring.vertices  = "offlinePrimaryVertices"

hltTOPmonitoring.btagAlgos = ['pfDeepCSVJetTags:probb', 'pfDeepCSVJetTags:probbb']
hltTOPmonitoring.workingpoint = 0.8484 # Medium wp

hltTOPmonitoring.HTdefinition = 'pt>30 & abs(eta)<2.5'
hltTOPmonitoring.leptJetDeltaRmin = 0.4
hltTOPmonitoring.bJetMuDeltaRmax  = 9999.
hltTOPmonitoring.bJetDeltaEtaMax  = 9999.

hltTOPmonitoring.numGenericTriggerEventPSet.andOr         = False
hltTOPmonitoring.numGenericTriggerEventPSet.andOrHlt      = True # True:=OR; False:=AND
hltTOPmonitoring.numGenericTriggerEventPSet.hltInputTag   = "TriggerResults::HLT"
hltTOPmonitoring.numGenericTriggerEventPSet.errorReplyHlt = False
hltTOPmonitoring.numGenericTriggerEventPSet.verbosityLevel = 0

hltTOPmonitoring.denGenericTriggerEventPSet.andOr         = False
hltTOPmonitoring.denGenericTriggerEventPSet.andOrHlt      = True # True:=OR; False:=AND
hltTOPmonitoring.denGenericTriggerEventPSet.hltInputTag   = "TriggerResults::HLT"
hltTOPmonitoring.denGenericTriggerEventPSet.errorReplyHlt = False
hltTOPmonitoring.denGenericTriggerEventPSet.dcsInputTag   = "scalersRawToDigi"
hltTOPmonitoring.denGenericTriggerEventPSet.dcsPartitions = [24, 25, 26, 27, 28, 29] # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltTOPmonitoring.denGenericTriggerEventPSet.andOrDcs      = False
hltTOPmonitoring.denGenericTriggerEventPSet.errorReplyDcs = True
hltTOPmonitoring.denGenericTriggerEventPSet.verbosityLevel = 0

hltTOPmonitoring.MHTdefinition = 'pt>30 & abs(eta)<2.5'
hltTOPmonitoring.MHTcut = -1
hltTOPmonitoring.invMassUppercut = -1.0
hltTOPmonitoring.invMassLowercut = -1.0
hltTOPmonitoring.oppositeSignMuons = False
hltTOPmonitoring.invMassCutInAllMuPairs = False
