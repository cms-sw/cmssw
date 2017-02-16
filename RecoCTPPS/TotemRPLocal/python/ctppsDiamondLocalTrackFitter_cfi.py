import FWCore.ParameterSet.Config as cms

ctppsDiamondLocalTrack = cms.EDProducer("CTPPSDiamondLocalTrackFitter",
    verbosity = cms.int32(0),
    recHitsTag = cms.InputTag("ctppsDiamondRecHits"),
    trackingAlgorithmParams = cms.PSet(
        threshold = cms.double(1.5),
        threshold_from_maximum = cms.double(0.5),
        resolution = cms.double(0.05), # in mm
        sigma = cms.double(0.1), # see below
        start_from_x_mm = cms.double(-0.5), # in mm
	stop_at_x_mm = cms.double(19.5), # in mm
	pixel_efficiency_function = cms.string("(TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4"),
    ),
)


#
#	"(TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4" Precise
#	"(x>[0]-0.5*[1])*(x<[0]+0.5*[1])+((x-[0]+0.5*[1]+[2])/[2])*(x>[0]-0.5*[1]-[2])*(x<[0]-0.5*[1])+(2-(x-[0]-0.5*[1]+[2])/[2])*(x>[0]+0.5*[1])*(x<[0]+0.5*[1]+[2])" Fast
#	[0]: centre of pad
#	[1]: width of pad
#	[2]: sigma: distance between efficeincy ~100 -> 0 outside width
#

# Legacy
#"(1/(1+exp(-(x-[0]+0.5*[1])/[2])))*(1/(1+exp((x-[0]-0.5*[1])/[2])))"