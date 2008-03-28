import FWCore.ParameterSet.Config as cms

from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Configuration.JetMET.CaloConditions_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
caloReco = cms.Sequence(calolocalreco)

