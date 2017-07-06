import FWCore.ParameterSet.Config as cms
process = cms.Process("GeometryTest")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# no events to process
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

# plotter for functions in sector 45
process.ctppsPlotOpticalFunctions_45 = cms.EDAnalyzer("CTPPSPlotOpticalFunctions",
    opticsFile = cms.string("../data/2016_preTS2/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root"),

    opticsObjects = cms.vstring(
        "ip5_to_station_150_h_1_lhcb2",
        "ip5_to_station_150_h_2_lhcb2"
      ),

    # in m
    vtx0_y_45 = cms.double(300E-6),
    vtx0_y_56 = cms.double(200E-6),

    # in rad
    half_crossing_angle_45 = cms.double(+179.394E-6),
    half_crossing_angle_56 = cms.double(+191.541E-6),

    outputFile = cms.string("optical_functions_45.root")
)

# plotter for functions in sector 56
process.ctppsPlotOpticalFunctions_56 = cms.EDAnalyzer("CTPPSPlotOpticalFunctions",
    opticsFile = cms.string("../data/2016_preTS2/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root"),

    opticsObjects = cms.vstring(
        "ip5_to_station_150_h_1_lhcb1",
        "ip5_to_station_150_h_2_lhcb1"
      ),

    # in m
    vtx0_y_45 = cms.double(300E-6),
    vtx0_y_56 = cms.double(200E-6),

    # in rad
    half_crossing_angle_45 = cms.double(+179.394E-6),
    half_crossing_angle_56 = cms.double(+191.541E-6),

    outputFile = cms.string("optical_functions_56.root")
)

process.p = cms.Path(
    process.ctppsPlotOpticalFunctions_45
    * process.ctppsPlotOpticalFunctions_56
)
