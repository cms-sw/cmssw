
import FWCore.ParameterSet.Config as cms
#############################################################################
## Temporary due to bad naming of the jet algorithm in correction modules  ##
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFCHSL1Offset, ak4PFCHSL1Fastjet, ak4PFCHSL2Relative, ak4PFCHSL3Absolute, ak4PFCHSResidual, ak4PFCHSL2L3, ak4PFCHSL2L3Residual
ak4PFCHSL1Offset.algorithm = 'AK4PFchs'
ak4PFCHSL1Fastjet.algorithm = 'AK4PFchs'
ak4PFCHSL2Relative.algorithm = 'AK4PFchs'
ak4PFCHSL3Absolute.algorithm = 'AK4PFchs'
ak4PFCHSResidual.algorithm = 'AK4PFchs'

topDQMak5PFCHSL1Offset = ak4PFCHSL1Offset.clone()
topDQMak5PFCHSL1Fastjet = ak4PFCHSL1Fastjet.clone()
topDQMak5PFCHSL2Relative = ak4PFCHSL2Relative.clone()
topDQMak5PFCHSL3Absolute = ak4PFCHSL3Absolute.clone()
topDQMak5PFCHSResidual = ak4PFCHSResidual.clone()

topDQMak5PFCHSL2L3 = ak4PFCHSL2L3.clone(correctors = cms.vstring('topDQMak5PFCHSL2Relative','topDQMak5PFCHSL3Absolute'))
topDQMak5PFCHSL2L3Residual = ak4PFCHSL2L3Residual.clone(correctors = cms.vstring('topDQMak5PFCHSL2Relative','topDQMak5PFCHSL3Absolute','topDQMak5PFCHSResidual'))
#############################################################################
