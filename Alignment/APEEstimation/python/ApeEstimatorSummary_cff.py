import os

import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.ApeEstimatorSummary_cfi import *



ApeEstimatorSummaryBaseline = ApeEstimatorSummary.clone(
    setBaseline = True,
    apeWeight = "entriesOverSigmaX2",
    #sigmaFactorFit = 2.5,
)



ApeEstimatorSummaryIter = ApeEstimatorSummary.clone(
    #setBaseline = False,
    apeWeight = "entriesOverSigmaX2",
    #sigmaFactorFit = 2.5,
    correctionScaling = 0.6,
)




