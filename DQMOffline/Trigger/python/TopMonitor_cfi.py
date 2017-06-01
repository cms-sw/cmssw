import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltTOPmonitoring = topMonitoring.clone()
hltTOPmonitoring.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/default/')
hltTOPmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.int32 (  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltTOPmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.int32 (  60   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltTOPmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.int32 (  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
hltTOPmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.int32 (  24  ),
  xmin  = cms.double( -2.4 ),
  xmax  = cms.double(  2.4 ),
)
hltTOPmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.int32 (   60  ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  600  ),
)

#MET and HT binning
hltTOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200)
hltTOPmonitoring.histoPSet.HTBinning  = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700)
#Eta binning
hltTOPmonitoring.histoPSet.eleEtaBinning = cms.vdouble(-2.4,-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1,2.4)
hltTOPmonitoring.histoPSet.jetEtaBinning = cms.vdouble(-2.4,-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1,2.4)
hltTOPmonitoring.histoPSet.muEtaBinning  = cms.vdouble(-2.4,-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1,2.4)
#pt binning
hltTOPmonitoring.histoPSet.elePtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,60,80,120,200,400)
hltTOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,3,5,8,15,20,25,30,40,50,60,80,120,200,400)
hltTOPmonitoring.histoPSet.muPtBinning  = cms.vdouble(0,3,5,8,15,20,25,30,40,50,60,80,120,200,400)
#Eta binning 2D
hltTOPmonitoring.histoPSet.eleEtaBinning2D = cms.vdouble(-2.4,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.4)
hltTOPmonitoring.histoPSet.jetEtaBinning2D = cms.vdouble(-2.4,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.4)
hltTOPmonitoring.histoPSet.muEtaBinning2D  = cms.vdouble(-2.4,-1.7,-1.2,-0.6,-0.3,0,0.3,0.6,1.2,1.7,2.4)
#pt binning 2D
hltTOPmonitoring.histoPSet.elePtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltTOPmonitoring.histoPSet.jetPtBinning2D = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
hltTOPmonitoring.histoPSet.muPtBinning2D  = cms.vdouble(0,15,20,30,40,60,80,100,200,400)
#HT and phi binning 2D
hltTOPmonitoring.histoPSet.HTBinning2D  = cms.vdouble(0,20,40,70,100,150,200,400,700)
hltTOPmonitoring.histoPSet.phiBinning2D = cms.vdouble(-3.1416,-2.5132,-1.8849,-1.2566,-0.6283,0,0.6283,1.2566,1.8849,2.5132,3.1416)


hltTOPmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltTOPmonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltTOPmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltTOPmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !

hltTOPmonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
hltTOPmonitoring.leptJetDeltaRmin = cms.double(0.4)

hltTOPmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltTOPmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltTOPmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltTOPmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltTOPmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltTOPmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltTOPmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltTOPmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltTOPmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltTOPmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltTOPmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

