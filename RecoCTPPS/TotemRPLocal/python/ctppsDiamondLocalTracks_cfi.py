import FWCore.ParameterSet.Config as cms

ctppsDiamondLocalTracks = cms.EDProducer("CTPPSDiamondLocalTrackFitter",
    verbosity = cms.int32(0),
    recHitsTag = cms.InputTag("ctppsDiamondRecHits"),
    trackingAlgorithmParams = cms.PSet(
        threshold = cms.double(1.5),
        thresholdFromMaximum = cms.double(0.5),
        resolution = cms.double(0.01), # in mm
        sigma = cms.double(0.1), # see below
        startFromX = cms.double(-0.5), # in mm
        stopAtX = cms.double(19.5), # in mm
        pixelEfficiencyFunction = cms.string("(TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4"),
        yPosition = cms.double(0.0),
        yWidth = cms.double(0.0),
    ),
)

#
# NOTE: pixelEfficiencyFunction can be defined as following:
#
#  Precise:
#    "(TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4"
#  Fast:
#    "(x>[0]-0.5*[1])*(x<[0]+0.5*[1])+((x-[0]+0.5*[1]+[2])/[2])*(x>[0]-0.5*[1]-[2])*(x<[0]-0.5*[1])+(2-(x-[0]-0.5*[1]+[2])/[2])*(x>[0]+0.5*[1])*(x<[0]+0.5*[1]+[2])"
#  Legacy:
#    "(1/(1+exp(-(x-[0]+0.5*[1])/[2])))*(1/(1+exp((x-[0]-0.5*[1])/[2])))"
#
#  with:
#    [0]: centre of pad
#    [1]: width of pad
#    [2]: sigma: distance between efficiency ~100 -> 0 outside width
#
