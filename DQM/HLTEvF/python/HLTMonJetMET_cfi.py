import FWCore.ParameterSet.Config as cms

hltMonJetMET = cms.EDAnalyzer("HLTMon",
   outputFile = cms.untracked.string('./L1TDQM.root'),
   verbose = cms.untracked.bool(False),
   MonitorDaemon = cms.untracked.bool(True),
   reqNum = cms.uint32(1),
   DaqMonitorBEInterface = cms.untracked.bool(True),
   filters = cms.VPSet(
       cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet15U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
     ),
       cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hlt1jet15U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(95),
     ),
       cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet30U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
     ),
       cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hlt1jet30U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(95),
     )),
disableROOToutput = cms.untracked.bool(True)


)
