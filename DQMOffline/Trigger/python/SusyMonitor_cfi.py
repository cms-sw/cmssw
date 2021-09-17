import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltSUSYmonitoring = topMonitoring.clone()
hltSUSYmonitoring.FolderName = cms.string('HLT/SUSY/default/')
hltSUSYmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltSUSYmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32(  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltSUSYmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32( 2500 ),
)
hltSUSYmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32(  60   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltSUSYmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32(  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
hltSUSYmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32(  24  ),
  xmin  = cms.double( -2.4 ),
  xmax  = cms.double(  2.4 ),
)
hltSUSYmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32(   60  ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  600  ),
)
# Marina
hltSUSYmonitoring.histoPSet.csvPSet = cms.PSet(
  nbins = cms.uint32( 50 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 1.0  ),
)
#BTV
hltSUSYmonitoring.histoPSet.DRPSet = cms.PSet(
  nbins = cms.uint32( 60  ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 6.0 ),
)

#Suvankar
hltSUSYmonitoring.applyLeptonPVcuts = False
hltSUSYmonitoring.leptonPVcuts = cms.PSet(
  dxy = cms.double(   9999.   ),
  dz  = cms.double(   9999.   ),
)

hltSUSYmonitoring.histoPSet.invMassPSet = cms.PSet(
  nbins = cms.uint32( 40 ),
  xmin  = cms.double( 0.0 ),
  xmax  = cms.double( 80.0  ),
)
hltSUSYmonitoring.histoPSet.MHTPSet = cms.PSet(
 nbins = cms.uint32(   80  ),
 xmin  = cms.double(   60   ),
 xmax  = cms.double(  300  ),
)


#MET and HT binning
hltSUSYmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200)
hltSUSYmonitoring.histoPSet.HTBinning  = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700)
#Eta binning
hltSUSYmonitoring.histoPSet.eleEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltSUSYmonitoring.histoPSet.jetEtaBinning = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
hltSUSYmonitoring.histoPSet.muEtaBinning  = cms.vdouble(-2.4,-2.1,-1.5,-0.9,-0.3,0.,0.3,0.9,1.5,2.1,2.4)
#pt binning
hltSUSYmonitoring.histoPSet.elePtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
hltSUSYmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
hltSUSYmonitoring.histoPSet.muPtBinning  = cms.vdouble(0,5,10,20,30,40,50,70,100,200,400)
#Eta binning 2D
hltSUSYmonitoring.histoPSet.eleEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltSUSYmonitoring.histoPSet.jetEtaBinning2D = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
hltSUSYmonitoring.histoPSet.muEtaBinning2D  = cms.vdouble(-2.5,-1.5,-0.6,0.,0.6,1.5,2.5)
#pt binning 2D
hltSUSYmonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
hltSUSYmonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,20,30,50,100,200,400)
hltSUSYmonitoring.histoPSet.muPtBinning2D  = cms.vdouble(0,20,30,50,100,200,400)
#HT and phi binning 2D
hltSUSYmonitoring.histoPSet.HTBinning2D  = cms.vdouble(0,20,40,70,100,150,200,400,700)
hltSUSYmonitoring.histoPSet.phiBinning2D = cms.vdouble(-3.1416,-1.8849,-0.6283,0.6283,1.8849,3.1416)


hltSUSYmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltSUSYmonitoring.jets      = cms.InputTag("ak4PFJetsCHS") # ak4PFJets, ak4PFJetsCHS, pfJetsEI
hltSUSYmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltSUSYmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !
#Suvankar
hltSUSYmonitoring.vertices  = cms.InputTag("offlinePrimaryVertices")

# Marina
hltSUSYmonitoring.btagAlgos = ["pfCombinedSecondaryVertexV2BJetTags"]
hltSUSYmonitoring.workingpoint = cms.double(0.8484) # Medium

hltSUSYmonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
hltSUSYmonitoring.leptJetDeltaRmin = cms.double(0.4)

hltSUSYmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltSUSYmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltSUSYmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltSUSYmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltSUSYmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltSUSYmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltSUSYmonitoring.denGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltSUSYmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltSUSYmonitoring.denGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltSUSYmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltSUSYmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltSUSYmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltSUSYmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltSUSYmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

