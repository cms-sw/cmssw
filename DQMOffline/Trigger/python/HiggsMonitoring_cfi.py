import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltHIGmonitoring = topMonitoring.clone()
#hltHIGmonitoring.FolderName = cms.string('HLT/Higgs/default/')
hltHIGmonitoring.FolderName = cms.string('HLT/HIG/default/')
hltHIGmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltHIGmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32 (  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltHIGmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 2500 ),
)
hltHIGmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32 (  60   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltHIGmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32 (  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
hltHIGmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32 (  30  ),
  xmin  = cms.double( -3.0 ),
  xmax  = cms.double(  3.0 ),
)
hltHIGmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32 (   60  ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  600  ),
)

# Marina
hltHIGmonitoring.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32 ( 50 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 1.0  ),
)

hltHIGmonitoring.histoPSet.DRPSet = cms.PSet(
   nbins = cms.uint32 ( 60  ),
   xmin  = cms.double( 0.0 ),
   xmax  = cms.double( 6.0 ),
)

hltHIGmonitoring.applyleptonPVcuts = cms.bool(True)
hltHIGmonitoring.leptonPVcuts = cms.PSet(
  dxy = cms.double(   0.5  ),
  dz  = cms.double(   1.   ),
)
hltHIGmonitoring.histoPSet.invMassPSet = cms.PSet(
  nbins = cms.uint32( 40 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 80.0  ),
)
hltHIGmonitoring.histoPSet.MHTPSet = cms.PSet(
 nbins = cms.uint32(   80  ),
 xmin  = cms.double(   60   ),
 xmax  = cms.double(  300  ),
)


#MET and HT binning
hltHIGmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200)
hltHIGmonitoring.histoPSet.HTBinning  = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700)
#Eta binning
hltHIGmonitoring.histoPSet.eleEtaBinning = cms.vdouble(-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5)
hltHIGmonitoring.histoPSet.jetEtaBinning = cms.vdouble(-4.7,-3.2,-3.0,-2.5,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.5,3.0,3.2,4.7)
hltHIGmonitoring.histoPSet.muEtaBinning  = cms.vdouble(-2.4,-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1,2.4)
#pt binning
hltHIGmonitoring.histoPSet.elePtBinning = cms.vdouble(0,3,5,8,10,15,20,25,30,40,50,60,80,120,200,400)
hltHIGmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,3,5,8,10,15,20,25,30,40,50,60,80,120,200,400)
hltHIGmonitoring.histoPSet.muPtBinning  = cms.vdouble(0,3,5,8,10,15,20,25,30,40,50,60,80,120,200,400)
#Eta binning 2D
hltHIGmonitoring.histoPSet.eleEtaBinning2D = cms.vdouble(-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5)
hltHIGmonitoring.histoPSet.jetEtaBinning2D = cms.vdouble(-4.7,-3.2,-3.0,-2.5,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.5,3.0,3.2,4.7)
hltHIGmonitoring.histoPSet.muEtaBinning2D  = cms.vdouble(-2.4,-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1,2.4)

#pt binning 2D
hltHIGmonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltHIGmonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltHIGmonitoring.histoPSet.muPtBinning2D  = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
#HT and phi binning 2D
hltHIGmonitoring.histoPSet.HTBinning2D  = cms.vdouble(0,20,40,70,100,150,200,400,700)
hltHIGmonitoring.histoPSet.phiBinning2D = cms.vdouble(-3.1416,-2.5132,-1.8849,-1.2566,-0.6283,0,0.6283,1.2566,1.8849,2.5132,3.1416)


hltHIGmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltHIGmonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltHIGmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltHIGmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !
hltHIGmonitoring.vertices  = cms.InputTag("offlinePrimaryVertices")

hltHIGmonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
hltHIGmonitoring.leptJetDeltaRmin = cms.double(0.4)
hltHIGmonitoring.eleSelection =  cms.string('pt > 7. && abs(eta) < 2.5') 
hltHIGmonitoring.muoSelection =  cms.string('pt > 5 &&  abs(eta) < 2.4 && (isGlobalMuon || (isTrackerMuon && numberOfMatches>0)) && muonBestTrackType != 2')
hltHIGmonitoring.vertexSelection = cms.string('!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2')

hltHIGmonitoring.nmuons     = cms.uint32(0)
hltHIGmonitoring.nelectrons = cms.uint32(0)
hltHIGmonitoring.njets      = cms.uint32(0)


hltHIGmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltHIGmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltHIGmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltHIGmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltHIGmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltHIGmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltHIGmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltHIGmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltHIGmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltHIGmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltHIGmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltHIGmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltHIGmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltHIGmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

