import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltTOPmonitoring = topMonitoring.clone()
hltTOPmonitoring.FolderName = cms.string('HLT/TOP/default/')
hltTOPmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltTOPmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32(  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltTOPmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32(  60   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltTOPmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 2500 ),
)
hltTOPmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32(  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
hltTOPmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32(  24  ),
  xmin  = cms.double( -2.4 ),
  xmax  = cms.double(  2.4 ),
)
hltTOPmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32(   60  ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  600  ),
)
# Marina
hltTOPmonitoring.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 50 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 1.0  ),
)
#BTV
hltTOPmonitoring.histoPSet.DRPSet = cms.PSet(
  nbins = cms.uint32( 60  ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 6.0 ),
)

#Suvankar
hltTOPmonitoring.applyleptonPVcuts = cms.bool(False)
hltTOPmonitoring.leptonPVcuts = cms.PSet(
  dxy = cms.double(   9999.   ),
  dz  = cms.double(   9999.   ),
)
#george
hltTOPmonitoring.histoPSet.invMassPSet = cms.PSet(
  nbins = cms.uint32( 40 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 80.0  ),
)
hltTOPmonitoring.histoPSet.MHTPSet = cms.PSet(
 nbins = cms.uint32(   80  ),
 xmin  = cms.double(   60   ),
 xmax  = cms.double(  300  ),
)


hltTOPmonitoring.enablePhotonPlot = cms.bool(False)
hltTOPmonitoring.enableMETplot = cms.bool(False)

#MET and HT binning
hltTOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200)
hltTOPmonitoring.histoPSet.HTBinning  = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700)
#Eta binning
hltTOPmonitoring.histoPSet.eleEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltTOPmonitoring.histoPSet.jetEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltTOPmonitoring.histoPSet.muEtaBinning  = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
#pt binning
hltTOPmonitoring.histoPSet.elePtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
hltTOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
hltTOPmonitoring.histoPSet.muPtBinning  = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
#Eta binning 2D
hltTOPmonitoring.histoPSet.eleEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltTOPmonitoring.histoPSet.jetEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltTOPmonitoring.histoPSet.muEtaBinning2D  = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltTOPmonitoring.histoPSet.phoEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
#pt binning 2D
hltTOPmonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
hltTOPmonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
hltTOPmonitoring.histoPSet.muPtBinning2D  = cms.vdouble(0,20,30,50,100,200,400)
hltTOPmonitoring.histoPSet.phoPtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
#HT and phi binning 2D
hltTOPmonitoring.histoPSet.HTBinning2D  = cms.vdouble(0,20,40,70,100,150,200,400,700)
hltTOPmonitoring.histoPSet.phiBinning2D = cms.vdouble(-3.1416,-1.8849,-0.6283,0.6283,1.8849,3.1416)


hltTOPmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltTOPmonitoring.jets      = cms.InputTag("ak4PFJetsCHS") # ak4PFJets, ak4PFJetsCHS, pfJetsEI
hltTOPmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltTOPmonitoring.elecID    = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-Fall17-94X-V1-tight") #Electron ID

hltTOPmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !
hltTOPmonitoring.photons   = cms.InputTag("photons") #reco::Photon 
#Suvankar
hltTOPmonitoring.vertices  = cms.InputTag("offlinePrimaryVertices")

# Marina
hltTOPmonitoring.btagalgo  = cms.InputTag("pfCombinedSecondaryVertexV2BJetTags")
hltTOPmonitoring.workingpoint     = cms.double(0.8484) # Medium

hltTOPmonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
hltTOPmonitoring.leptJetDeltaRmin = cms.double(0.4)
hltTOPmonitoring.bJetMuDeltaRmax  = cms.double(9999.)
hltTOPmonitoring.bJetDeltaEtaMax  = cms.double(9999.)

hltTOPmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltTOPmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltTOPmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltTOPmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltTOPmonitoring.denGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltTOPmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltTOPmonitoring.denGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltTOPmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltTOPmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltTOPmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltTOPmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltTOPmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)
#george
hltTOPmonitoring.MHTdefinition = cms.string('pt>30 & abs(eta)<2.5')
hltTOPmonitoring.MHTcut = cms.double(-1)
hltTOPmonitoring.invMassUppercut=cms.double(-1.0)
hltTOPmonitoring.invMassLowercut=cms.double(-1.0)
hltTOPmonitoring.oppositeSignMuons=cms.bool(False)
hltTOPmonitoring.invMassCutInAllMuPairs=cms.bool(False)
