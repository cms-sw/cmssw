import FWCore.ParameterSet.Config as cms

# Standard Cambridge/Kt jets parameters
# $Id: CambridgeJetParameters.cfi,v 1.1 2007/08/02 21:58:22 fedor Exp $
CambridgeJetParameters = cms.PSet(
    #possible Strategies: "Best","N2Plain","N2Tiled","N2MinHeapTiled","NlnN","NlnNCam"
    Strategy = cms.string('Best')
)

