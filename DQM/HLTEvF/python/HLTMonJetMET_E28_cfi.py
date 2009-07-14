import FWCore.ParameterSet.Config as cms

#path names from from http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/browser/
#commisssioning09/MWGR/MWGR26/HLTallDQM

hltMonJetMET = cms.EDAnalyzer("HLTMonJetMETDQMSource",
   outputFile = cms.untracked.string('./L1TDQM.root'),
   verbose = cms.untracked.bool(False),
   MonitorDaemon = cms.untracked.bool(True),
   reqNum = cms.uint32(1),
   DaqMonitorBEInterface = cms.untracked.bool(True),
   filters = cms.VPSet(      

     cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sL1Jet6U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
           theSubDir = cms.string("L1Jet"),
      ),
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sL1Jet6U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(85),
           theSubDir = cms.string("L1Jet"),
      ),cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sL1Jet6U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(86),
           theSubDir = cms.string("L1Jet"),
      ),
     
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet15U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
           theSubDir = cms.string("L1Jet"),
      ),
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet15U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(85),
           theSubDir = cms.string("L1Jet"),
      ),cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet15U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(86),
           theSubDir = cms.string("L1Jet"),
      ),
     
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet30U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
           theSubDir = cms.string("L1Jet"),
      ),
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet30U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(85),
           theSubDir = cms.string("L1Jet"),
      ),cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet30U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(86),
           theSubDir = cms.string("L1Jet"),
      ),
     

      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet50U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
           theSubDir = cms.string("L1Jet"),
      ),
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet50U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(85),
           theSubDir = cms.string("L1Jet"),
      ),cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sJet50U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(86),
           theSubDir = cms.string("L1Jet"),
      ),
     

      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sDiJetAve15U8E29","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
           theSubDir = cms.string("L1DiJet"),
      ),
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sDiJetAve15U8E29","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(85),
           theSubDir = cms.string("L1DiJet"),
      ),cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sDiJetAve15U8E29","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(86),
           theSubDir = cms.string("L1DiJet"),
      ),
     

      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sDiJetAve30U8E29","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(84),
           theSubDir = cms.string("L1DiJet"),
      ),
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sDiJetAve30U8E29","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(85),
           theSubDir = cms.string("L1DiJet"),
      ),cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sDiJetAve30U8E29","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(86),
           theSubDir = cms.string("L1DiJet"),
      ),
     

      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sL1MET20","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(87),
           theSubDir = cms.string("L1MET"),
      ),       
 
     cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sMET35","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(87),
           theSubDir = cms.string("L1MET"),
      ),       
 
     cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltL1sMET100","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(87),
           theSubDir = cms.string("L1MET"),
      ),       
 

     cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hlt1jet15U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("Jet"),
      ),       
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hlt1jet30U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("Jet"),
      ),       
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hlt1jet50U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("Jet"),
      ),       
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltDiJetAve15U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("DiJet"),
      ),       
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltDiJetAve30U","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("DiJet"),
      ),       
      cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hltPreL1MET20","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("MET"),
      ),       

    cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hlt1MET35","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("MET"),
      ),       

     cms.PSet(
           PlotBounds = cms.vdouble(0.0, 0.0),
           HLTCollectionLabels = cms.InputTag("hlt1MET100","","HLT"),
           IsoCollections = cms.VInputTag(cms.InputTag("none")),
           theHLTOutputTypes = cms.uint32(97),
           theSubDir = cms.string("MET"),
      ),       



      disableROOToutput = cms.untracked.bool(True)
   )

)

