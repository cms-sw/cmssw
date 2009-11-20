import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOnline_cfi.py,v 1.5 2009/11/07 21:31:37 rekovic Exp $
hltResultsOn = cms.EDFilter("FourVectorHLTOnline",
    dirname = cms.untracked.string("HLT/FourVector/source"),
    plotAll = cms.untracked.bool(False),

    ptMin = cms.untracked.double(0.0),
    ptMax = cms.untracked.double(100.0),
    Nbins = cms.untracked.uint32(10000),

    oneOverPtMin = cms.untracked.double(0.0),
    oneOverPtMax = cms.untracked.double(1.0),
    NbinsOneOver = cms.untracked.uint32(10000),
    paths = cms.VPSet(
# single jet triggers
             cms.PSet(
              pathname = cms.string("HLT_Jet50"),
              denompathname = cms.string("HLT_Jet30"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet110"),
              denompathname = cms.string("HLT_Jet50"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_L2Mu3"),
              denompathname = cms.string("HLT_L1Mu"),
             )
            ),
     SpecialPaths = cms.vstring(
            'HLT_L1MuOpen',
            'HLT_MinBias', 
            'HLT_L1_BPTX', 
            'HLT_MET100'
      ),
                          
     # this is I think MC and CRUZET4
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    processname = cms.string("HLT")

    # this is data (CRUZET I or II best guess)
    #triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","FU"),
    #triggerResultsLabel = cms.InputTag("TriggerResults","","FU"),
    #processname = cms.string("FU")

 )
