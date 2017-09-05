import FWCore.ParameterSet.Config as cms

ecalpedestalPCL =cms.EDAnalyzer('ECALpedestalPCLworker',
                                BarrelDigis=cms.InputTag('ecalDigis','ebDigis'),
                                EndcapDigis=cms.InputTag('ecalDigis','eeDigis'),
                                bstRecord  =cms.InputTag('tcdsDigis','bstRecord'),
                                requireStableBeam = cms.bool(True),
                                pedestalSamples=cms.uint32(2),   # number of presamples to be used for pedestal determination
                                checkSignal = cms.bool(False),   # whether or not to exclude digis containing a signal
                                sThresholdEB  = cms.uint32(10),  # threshold to define a digi as containing signal
                                sThresholdEE  = cms.uint32(15),
                                dqmDir      = cms.string('AlCaReco/EcalPedestalsPCL'), 
                                dynamicBooking = cms.bool(True), # use old pedestal to book histograms (central bin)
                                fixedBookingCenterBin = cms.int32(200), # if dynamicBooking = false, use this as center bin 
                                nBins       = cms.int32(40)      # number of bins per histogram
)
