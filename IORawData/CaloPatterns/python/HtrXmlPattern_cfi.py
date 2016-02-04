import FWCore.ParameterSet.Config as cms

htr_xml = cms.EDAnalyzer("HtrXmlPattern",
    #for non-existent electronics.
    presamples_per_event = cms.untracked.int32(4),
    sets_to_show = cms.untracked.int32(0), ##For a non-negative integer, dump an amount of data to stdout

    single_XML_file = cms.untracked.bool(True), ##When true, all patterns are placed in a single

    write_XML = cms.untracked.bool(True), ##When true, XML files containing the pattern data are produced.

    #XML file (otherwise one file per channel).
    file_tag = cms.untracked.string('example'),
    #that is proportional to this number.
    #For any negative number, dump all available data to stdout.
    #Non-zero values are typically used for debugging.
    show_errors = cms.untracked.bool(True),
    samples_per_event = cms.untracked.int32(10), ##Keep up to this many total samples

    #(including pre-samples) per event.
    write_root_file = cms.untracked.bool(True),
    user_output_directory = cms.untracked.string('/tmp') ##user_output_directory and will contain all produced files.

)


