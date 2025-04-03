import FWCore.ParameterSet.Config as cms

#Parameterset for the hadronic shower trigger for Run-3
showerPSet = cms.PSet(
    ## what kind of shower triggers the logic?
    ## 0: cathode-only (TMB/OTMB)
    ## 1: anode-only (from ALCT board)
    ## 2: cathode or anode showers
    ##    loose -> 'loose anode or loose cathode'
    ##    nominal -> 'nominal anode or nominal cathode'
    ##    tight -> 'tight anode or tight cathode'
    ## 3: cathode and anode showers
    ##    loose -> 'loose anode and loose cathode'
    ##    nominal -> 'nominal anode and nominal cathode'
    ##    tight -> 'tight anode and tight cathode'
    source  = cms.vuint32(
	# ME1/1
	3,
	# ME1/2
	1,
	# ME1/3
	1,
	# ME2/1
	3,
	# ME2/2
	1,
	# ME3/1
	3,
	# ME3/2
	1,
	# ME4/1
	3,
	# ME4/2
	1
	),

    ## settings for cathode showers (counting CSCComparatorDigi)
    cathodeShower = cms.PSet(
        ## {loose, nominal, tight} thresholds for hit counters
        ## loose ~ 0.75 kHz
        ## nominal ~ 0.5  kHz
        ## tight ~ 0.25 kHz
	## 10000 means to disable cathode HMT for this chamber type
        showerThresholds = cms.vuint32(
            # ME1/1
            100, 100, 100,
            # ME1/2
            10000, 10000, 10000,
            # ME1/3
            10000, 10000, 10000,
            # ME2/1
            14, 33, 35,
            # ME2/2
            10000, 10000, 10000,
            # ME3/1
            12, 31, 33,
            # ME3/2
            10000, 10000, 10000,
            # ME4/1
            14, 34, 36,
            # ME4/2
            10000, 10000, 10000
        ),
        showerNumTBins = cms.uint32(3),# 3BX for cathode HMT
        minLayersCentralTBin = cms.uint32(5),
	## peack check feature is not implemented in firmware
	## plan to upgrade in future
	peakCheck = cms.bool(False),
    ),
    ## settings for anode showers (counting CSCWireDigi)
    anodeShower = cms.PSet(
        ## {loose, nominal, tight} thresholds for hit counters
        showerThresholds = cms.vuint32(
            # ME1/1
            140, 140, 140,
            # ME1/2
            140, 140, 140,
            # ME1/3
            7, 14, 18,
            # ME2/1
            23, 56, 58,
            # ME2/2
            12, 28, 32,
            # ME3/1
            21, 55, 57,
            # ME3/2
            12, 26, 34,
            # ME4/1
            25, 62, 64,
            # ME4/2
            12, 27, 31
        ),
        showerNumTBins = cms.uint32(1),# 1BX for anode HMT
        minLayersCentralTBin = cms.uint32(5),
    )
)
