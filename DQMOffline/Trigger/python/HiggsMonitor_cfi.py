import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltHiggsmonitoring = topMonitoring.clone()
hltHiggsmonitoring.FolderName = cms.string('HLT/Higgs/default/')
hltHiggsmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32(  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltHiggsmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32(  100   ), #60
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  1000  ), #300
)
hltHiggsmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32(  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
hltHiggsmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32(  24  ),
  xmin  = cms.double( -2.4 ),
  xmax  = cms.double(  2.4 ),
)
hltHiggsmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32(   100  ), #60
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  1000  ), #600
)
# Marina
hltHiggsmonitoring.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 50 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 1.0  ),
)
#BTV
hltHiggsmonitoring.histoPSet.DRPSet = cms.PSet(
  nbins = cms.uint32( 60  ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 6.0 ),
)

#Suvankar
hltHiggsmonitoring.applyleptonPVcuts = cms.bool(False)
hltHiggsmonitoring.leptonPVcuts = cms.PSet(
  dxy = cms.double(   9999.   ),
  dz  = cms.double(   9999.   ),
)

#MET and HT binning
hltHiggsmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200)
hltHiggsmonitoring.histoPSet.HTBinning  = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700)
#Eta binning
hltHiggsmonitoring.histoPSet.eleEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltHiggsmonitoring.histoPSet.jetEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltHiggsmonitoring.histoPSet.muEtaBinning  = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
#pt binning
#hltHiggsmonitoring.histoPSet.elePtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
#hltHiggsmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
#hltHiggsmonitoring.histoPSet.muPtBinning  = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
hltHiggsmonitoring.histoPSet.elePtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,60,80,120,200,400,700)
hltHiggsmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,70,100,150,200,400,700,1000,1500)
hltHiggsmonitoring.histoPSet.muPtBinning = cms.vdouble(0,3,5,7,10,15,20,30,40,50,70,100,150,200,400,700)
#Eta binning 2D
hltHiggsmonitoring.histoPSet.eleEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltHiggsmonitoring.histoPSet.jetEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltHiggsmonitoring.histoPSet.muEtaBinning2D  = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
#pt binning 2D
#hltHiggsmonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
#hltHiggsmonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
#hltHiggsmonitoring.histoPSet.muPtBinning2D  = cms.vdouble(0,20,30,50,100,200,400)
hltHiggsmonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltHiggsmonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltHiggsmonitoring.histoPSet.muPtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
#HT and phi binning 2D
hltHiggsmonitoring.histoPSet.HTBinning2D  = cms.vdouble(0,20,40,70,100,150,200,400,700)
hltHiggsmonitoring.histoPSet.phiBinning2D = cms.vdouble(-3.1416,-1.8849,-0.6283,0.6283,1.8849,3.1416)


hltHiggsmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltHiggsmonitoring.jets      = cms.InputTag("ak4PFJetsCHS") # ak4PFJets, ak4PFJetsCHS, pfJetsEI
hltHiggsmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltHiggsmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !
#Suvankar
hltHiggsmonitoring.vertices  = cms.InputTag("offlinePrimaryVertices")

# Marina
hltHiggsmonitoring.btagalgo  = cms.InputTag("pfCombinedSecondaryVertexV2BJetTags")
hltHiggsmonitoring.workingpoint     = cms.double(0.92) # tight

hltHiggsmonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
#hltHiggsmonitoring.leptJetDeltaRmin = cms.double(0.4) # MuonJet dRcone

#always monitor CSV score for one jet if set DeltaRmin = 0.0 and WP to -1 
#hltHiggsmonitoring.nbjets = cms.uint32(1)
#hltHiggsmonitoring.bjetSelection = cms.string('pt>30 & abs(eta)<2.4')

hltHiggsmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltHiggsmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltHiggsmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::reHLT" ) #change to HLT for PR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
hltHiggsmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltHiggsmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltHiggsmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltHiggsmonitoring.denGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltHiggsmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::reHLT" )  #change to HLT for PR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
hltHiggsmonitoring.denGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltHiggsmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltHiggsmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltHiggsmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltHiggsmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltHiggsmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

