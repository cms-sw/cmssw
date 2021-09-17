import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltBTVmonitoring = topMonitoring.clone()
hltBTVmonitoring.FolderName = cms.string('HLT/BTV/default/')
hltBTVmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltBTVmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32(  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltBTVmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 2500 ),
)
hltBTVmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32(  100  ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  1000 ),
)
hltBTVmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32(  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
hltBTVmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32(  24  ),
  xmin  = cms.double( -2.4 ),
  xmax  = cms.double(  2.4 ),
)
hltBTVmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32(  100  ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  1000 ),
)
hltBTVmonitoring.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 20 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 1.0  ),
)
hltBTVmonitoring.histoPSet.DRPSet = cms.PSet(
  nbins = cms.uint32( 60  ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 6.0 ),
)
hltBTVmonitoring.histoPSet.invMassPSet = cms.PSet(
  nbins = cms.uint32( 40 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 80.0  ),
)
hltBTVmonitoring.histoPSet.MHTPSet = cms.PSet(
 nbins = cms.uint32(   80  ),
 xmin  = cms.double(   60   ),
 xmax  = cms.double(  300  ),
)

#MET and HT binning
hltBTVmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200)
hltBTVmonitoring.histoPSet.HTBinning  = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700)
#Eta binning
hltBTVmonitoring.histoPSet.eleEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltBTVmonitoring.histoPSet.jetEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltBTVmonitoring.histoPSet.muEtaBinning  = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
#pt binning
hltBTVmonitoring.histoPSet.elePtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,60,80,120,200,400,700)
hltBTVmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,70,100,150,200,400,700,1000,1500,3000)
hltBTVmonitoring.histoPSet.muPtBinning  = cms.vdouble(0,3,5,7,10,15,20,30,40,50,70,100,150,200,400,700)
#Eta binning 2D
hltBTVmonitoring.histoPSet.eleEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltBTVmonitoring.histoPSet.jetEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltBTVmonitoring.histoPSet.muEtaBinning2D  = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
#pt binning 2D
hltBTVmonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltBTVmonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltBTVmonitoring.histoPSet.muPtBinning2D  = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
#HT and phi binning 2D
hltBTVmonitoring.histoPSet.HTBinning2D  = cms.vdouble(0,20,40,70,100,150,200,400,700)
hltBTVmonitoring.histoPSet.phiBinning2D = cms.vdouble(-3.1416,-1.8849,-0.6283,0.6283,1.8849,3.1416)

hltBTVmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltBTVmonitoring.jets      = cms.InputTag("ak4PFJetsCHS") # ak4PFJets, ak4PFJetsCHS, pfJetsEI
hltBTVmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltBTVmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !

hltBTVmonitoring.btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"]
hltBTVmonitoring.workingpoint = cms.double(-1.) #no cut applied

hltBTVmonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
hltBTVmonitoring.leptJetDeltaRmin = cms.double(0.0)
hltBTVmonitoring.bJetMuDeltaRmax  = cms.double(9999.)
hltBTVmonitoring.bJetDeltaEtaMax  = cms.double(9999.)
#always monitor CSV score for one jet
hltBTVmonitoring.nbjets = cms.uint32(1)
hltBTVmonitoring.bjetSelection = cms.string('pt>30 & abs(eta)<2.4')

hltBTVmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltBTVmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltBTVmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltBTVmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltBTVmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltBTVmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltBTVmonitoring.denGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltBTVmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltBTVmonitoring.denGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltBTVmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltBTVmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltBTVmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltBTVmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltBTVmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

