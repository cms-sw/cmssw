import FWCore.ParameterSet.Config as cms

ootPileupCorrectionSerializer = cms.EDAnalyzer(
    "OOTPileupCorrectionSerializer",
    #
    # The output file will contain a generic serialization string
    # archive (GSSA). The contents will be subsequently compressed
    # by the database anyway, so we do not compress the archive here.
    outputFile = cms.string("ootPileupCorrection.gssa"),
    #
    # Correction specs. Each correction is specified by
    # a PSet. Each of these PSets must define string parameters
    # "Class", "name", and "category". Depending on the class,
    # other parameters may be necessary. The corrections are built
    # out of the parameter sets by the "parseOOTPileupCorrection"
    # finction defined inside "OOTPileupCorrectionSerializer.cc".
    #
    # Note that default category names in the HcalHitReconstructor
    # config files are "DataOOTPileupCorrections" for data corrections
    # and "MCOOTPileupCorrections" for MC corrections.
    #
    corrections = cms.VPSet(
        cms.PSet(
            Class = cms.string("DummyOOTPileupCorrection"),
            name = cms.string("Dummy"),
            category = cms.string("Example"),
            description = cms.string("This correction does not change anything"),
            scale = cms.double(1.0)
        ),
        cms.PSet(
            Class = cms.string("DummyOOTPileupCorrection"),
            name = cms.string("HF"),
            category = cms.string("OOTPileupCorrections"),
            description = cms.string("Multiplies all charge by a factor of 3"),
            scale = cms.double(3.0)
        ),
        cms.PSet(
            Class = cms.string("DummyOOTPileupCorrection"),
            name = cms.string("HO"),
            category = cms.string("OOTPileupCorrections"),
            description = cms.string("Multiplies all charge by a factor of 1/2"),
            scale = cms.double(0.5)
        ),
        cms.PSet(
            Class = cms.string("DummyOOTPileupCorrection"),
            name = cms.string("HBHE"),
            category = cms.string("OOTPileupCorrections"),
            description = cms.string("Multiplies all charge by a factor of 5"),
            scale = cms.double(5.0)
        )
    )
)
