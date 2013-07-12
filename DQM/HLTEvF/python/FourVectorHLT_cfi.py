# $Id: FourVectorHLT_cfi.py,v 1.5 2008/10/08 06:49:13 lorenzo Exp $
import FWCore.ParameterSet.Config as cms

hltResults = cms.EDAnalyzer("FourVectorHLT",
    plotAll = cms.untracked.bool(True),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    filters = cms.VPSet(),
    # data best guess
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)


