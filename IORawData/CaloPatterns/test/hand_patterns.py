import FWCore.ParameterSet.Config as cms

process = cms.Process("Patterns")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'DESIGN_3X_V14::All'
process.source = cms.Source("EmptySource")

process.htr_xml = cms.EDFilter("HtrXmlPattern",
    sets_to_show = cms.untracked.int32(0),
    show_errors = cms.untracked.bool(True),

    presamples_per_event = cms.untracked.int32(4),
    samples_per_event = cms.untracked.int32(10),

    write_root_file = cms.untracked.bool(True),
    XML_file_mode = cms.untracked.int32(3), #0=no-output; 1=one-file; 2=one-file-per-crate; 3=one-file-per-fiber
    file_tag = cms.untracked.string('example'),
    user_output_directory = cms.untracked.string('/tmp'),

    fill_by_hand = cms.untracked.bool(True),
    hand_pattern_number = cms.untracked.int32(3)
)

process.p = cms.Path(process.htr_xml)


