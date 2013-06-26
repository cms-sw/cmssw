import FWCore.ParameterSet.Config as cms

#PixelCPEParmErrorESProducer = cms.ESProducer("PixelCPEParmErrorESProducer",
#    # use new parameterization from CalibTracker/SiPixelErrorEstimation 
#    # errors are read in "PixelErrorParameterization.cc" from "data/residuals.dat"
#    UseNewParametrization = cms.bool(True),
#    ComponentName = cms.string('PixelCPEfromTrackAngle'),
#    # use Gaussian errors by default 
#    # if "UseSigma = false", then use RMS errors - not recommended 
#    UseSigma = cms.bool(True),
#    PixelErrorParametrization = cms.string('NOTcmsim'),
#    Alpha2Order = cms.bool(True),
#
#    # petar, for clusterProbability() from TTRHs
#    ClusterProbComputationFlag = cms.int32(0)
#)


