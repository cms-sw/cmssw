import FWCore.ParameterSet.Config as cms

# Standard kT jets parameters
# $Id: KtJetParameters.cfi,v 1.3 2007/08/02 21:58:22 fedor Exp $
KtJetParameters = cms.PSet(
    #possible Strategies: "Best","N2Plain","N2Tiled","N2MinHeapTiled","NlnN","NlnNCam"
    Strategy = cms.string('Best')
)

