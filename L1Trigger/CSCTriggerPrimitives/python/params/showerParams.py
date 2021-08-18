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
    source  = cms.uint32(0),

    ## settings for cathode showers (counting CSCComparatorDigi)
    cathodeShower = cms.PSet(
        ## {loose, nominal, tight} thresholds for hit counters
        ## loose ~ 0.75 kHz
        ## nominal ~ 0.5  kHz
        ## tight ~ 0.25 kHz
        showerThresholds = cms.vuint32(
            # ME1/1
            100, 100, 100,
            # ME1/2
            54, 55, 61,
            # ME1/3
            20, 20, 30,
            # ME2/1
            35, 35, 35,
            # ME2/2
            29, 29, 35,
            # ME3/1
            35, 35, 40,
            # ME3/2
            24, 25, 30,
            # ME4/1
            36, 40, 40,
            # ME4/2
            26, 30, 30
        ),
        showerMinInTBin = cms.uint32(6),
        showerMaxInTBin = cms.uint32(8),
        showerMinOutTBin = cms.uint32(2),
        showerMaxOutTBin = cms.uint32(5),
    ),
    ## settings for anode showers (counting CSCWireDigi)
    anodeShower = cms.PSet(
        ## {loose, nominal, tight} thresholds for hit counters
        showerThresholds = cms.vuint32(
            # ME1/1
            104, 105, 107,
            # ME1/2
            92, 100, 102,
            # ME1/3
            32, 33, 48,
            # ME2/1
            133, 134, 136,
            # ME2/2
            83, 84, 86,
            # ME3/1
            130, 131, 133,
            # ME3/2
            74, 80, 87,
            # ME4/1
            127, 128, 130,
            # ME4/2
            88, 89, 94
        ),
        showerMinInTBin = cms.uint32(8),
        showerMaxInTBin = cms.uint32(10),
        showerMinOutTBin = cms.uint32(4),
        showerMaxOutTBin = cms.uint32(7),
    )
)
