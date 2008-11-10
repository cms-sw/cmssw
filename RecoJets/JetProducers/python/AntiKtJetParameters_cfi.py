import FWCore.ParameterSet.Config as cms

# Standard kT jets parameters
# $Id: KtJetParameters_cfi.py,v 1.2 2008/04/21 03:28:56 rpw Exp $
AntiKtJetParameters = cms.PSet(
    #possible Strategies: "Best","N2Plain","N2Tiled","N2MinHeapTiled","NlnN","NlnNCam"
    Strategy = cms.string('Best')
)

