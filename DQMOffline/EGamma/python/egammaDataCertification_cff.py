import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonDataCertification_cfi import *

egammaDataCertificationTask = cms.Sequence(photonDataCertification)


