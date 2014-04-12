import FWCore.ParameterSet.Config as cms

qcdJetFilterStreamLo = cms.Sequence(~cms.SequencePlaceholder("qcdSingleJetFilterHi")*~cms.SequencePlaceholder("qcdSingleJetFilterMed")*cms.SequencePlaceholder("qcdSingleJetFilterLo"))

