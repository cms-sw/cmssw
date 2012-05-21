'''

Select PFTaus by rho-corrected PT

Author: Evan K. Friis, UW Madison

'''

import FWCore.ParameterSet.Config as cms

pfTauPtCutRhoCorrectedSelector = cms.EDFilter(
    "PFTauPtCutRhoCorrectedSelector",
    src = cms.InputTag("src"),
    srcRho = cms.InputTag("kt6PFJets", "rho"),
    effectiveArea = cms.double(0.2),
    minPt = cms.double(20),
    filter = cms.bool(True),
)

