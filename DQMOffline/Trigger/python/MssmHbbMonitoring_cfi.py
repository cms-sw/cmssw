import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

mssmHbbMonitoring = topMonitoring.clone()
#mssmHbbMonitoring.FolderName = cms.string('HLT/Higgs/default/')
mssmHbbMonitoring.FolderName = cms.string('HLT/HIG/default/')
mssmHbbMonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
mssmHbbMonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32(  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
mssmHbbMonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32(  100   ), #60
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  1000  ), #300
)
mssmHbbMonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 2500 ),
)
mssmHbbMonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32(  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
mssmHbbMonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32(  24  ),
  xmin  = cms.double( -2.4 ),
  xmax  = cms.double(  2.4 ),
)
mssmHbbMonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32(   100  ), #60
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  1000  ), #600
)
# Marina
mssmHbbMonitoring.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 50 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 1.0  ),
)
#BTV
mssmHbbMonitoring.histoPSet.DRPSet = cms.PSet(
  nbins = cms.uint32( 60  ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 6.0 ),
)

#Suvankar
mssmHbbMonitoring.applyleptonPVcuts = cms.bool(False)
mssmHbbMonitoring.leptonPVcuts = cms.PSet(
  dxy = cms.double(   9999.   ),
  dz  = cms.double(   9999.   ),
)
mssmHbbMonitoring.histoPSet.invMassPSet = cms.PSet(
  nbins = cms.uint32( 40 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 80.0  ),
)
mssmHbbMonitoring.histoPSet.MHTPSet = cms.PSet(
 nbins = cms.uint32(   80  ),
 xmin  = cms.double(   60   ),
 xmax  = cms.double(  300  ),
)

#MET and HT binning
mssmHbbMonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200)
mssmHbbMonitoring.histoPSet.HTBinning  = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700)
#Eta binning
mssmHbbMonitoring.histoPSet.eleEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
mssmHbbMonitoring.histoPSet.jetEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
mssmHbbMonitoring.histoPSet.muEtaBinning  = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
#pt binning
#mssmHbbMonitoring.histoPSet.elePtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
#mssmHbbMonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
#mssmHbbMonitoring.histoPSet.muPtBinning  = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
mssmHbbMonitoring.histoPSet.elePtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,60,80,120,200,400,700)
mssmHbbMonitoring.histoPSet.jetPtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,70,100,150,200,400,700,1000,1500)
mssmHbbMonitoring.histoPSet.muPtBinning = cms.vdouble(0,3,5,7,10,15,20,30,40,50,70,100,150,200,400,700)
#Eta binning 2D
mssmHbbMonitoring.histoPSet.eleEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
mssmHbbMonitoring.histoPSet.jetEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
mssmHbbMonitoring.histoPSet.muEtaBinning2D  = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
#pt binning 2D
#mssmHbbMonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
#mssmHbbMonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
#mssmHbbMonitoring.histoPSet.muPtBinning2D  = cms.vdouble(0,20,30,50,100,200,400)
mssmHbbMonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
mssmHbbMonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
mssmHbbMonitoring.histoPSet.muPtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
#HT and phi binning 2D
mssmHbbMonitoring.histoPSet.HTBinning2D  = cms.vdouble(0,20,40,70,100,150,200,400,700)
mssmHbbMonitoring.histoPSet.phiBinning2D = cms.vdouble(-3.1416,-1.8849,-0.6283,0.6283,1.8849,3.1416)


mssmHbbMonitoring.met       = cms.InputTag("pfMetEI") # pfMet
mssmHbbMonitoring.jets      = cms.InputTag("ak4PFJetsCHS") # ak4PFJets, ak4PFJetsCHS, pfJetsEI
mssmHbbMonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
mssmHbbMonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !
#Suvankar
mssmHbbMonitoring.vertices  = cms.InputTag("offlinePrimaryVertices")

# Marina
mssmHbbMonitoring.btagalgo         = cms.InputTag("pfCombinedSecondaryVertexV2BJetTags")
mssmHbbMonitoring.workingpoint     = cms.double(0.92) # tight


mssmHbbMonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
#mssmHbbMonitoring.leptJetDeltaRmin = cms.double(0.4) # MuonJet dRcone

#always monitor CSV score for one jet if set DeltaRmin = 0.0 and WP to -1 
#mssmHbbMonitoring.nbjets = cms.uint32(1)
#mssmHbbMonitoring.bjetSelection = cms.string('pt>30 & abs(eta)<2.4')

mssmHbbMonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
mssmHbbMonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
mssmHbbMonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" ) #change to HLT for PR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mssmHbbMonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
mssmHbbMonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

mssmHbbMonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
mssmHbbMonitoring.denGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
mssmHbbMonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )  #change to HLT for PR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mssmHbbMonitoring.denGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
mssmHbbMonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
mssmHbbMonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
mssmHbbMonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
mssmHbbMonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
mssmHbbMonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

