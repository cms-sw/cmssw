import FWCore.ParameterSet.Config as cms

process = cms.Process("ERRORSANALYZER")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.ErrorsAnalyzerModule = cms.EDAnalyzer(
    "ErrorsAnalyzer",

    InputFileName = cms.string("Zmumu_corrected.root"),
    MaxEvents = cms.int32(-1),

    # Function parameters
    ResolFitType = cms.int32(8),

    Parameters = cms.vdouble(-0.00450348, -0.0000000999966, 1.2576, 0.049718,
                             0.00043, 0.0041, 0.000028, 0.000077,
                             0.00011, 0.0018, -0.00000094, 0.000022),
    Errors = cms.vdouble(0.0019005, 0.0000000629359, 0.102255, 0.0120913,
                         0, 0, 0, 0,
                         0, 0, 0, 0),
    ErrorFactors = cms.vint32( 1, 1, 1, 1,
                               1, 1, 1, 1,
                               1, 1, 1, 1 ),

    OutputFileName = cms.string("test.root"),

    PtBins = cms.int32(50),
    EtaBins = cms.int32(100),

    Debug = cms.bool(False),
)

process.p1 = cms.Path(process.ErrorsAnalyzerModule)

