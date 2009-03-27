import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOffline_cfi.py,v 1.19 2009/02/26 22:30:17 berryhil Exp $
hltResults = cms.EDFilter("FourVectorHLTOffline",
    dirname = cms.untracked.string("HLT/FourVector/source"),
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    paths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_Mu11"),
              denompathname = cms.string("HLT_L1Jet15"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet30"),
              denompathname = cms.string("HLT_Mu3"),  
             ),
# single jet triggers
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet50"),  
#             )
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
