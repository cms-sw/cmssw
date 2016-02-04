import FWCore.ParameterSet.Config as cms

qcdJetFilterStreamMed = cms.Sequence(~cms.SequencePlaceholder("qcdSingleJetFilterHi")*cms.SequencePlaceholder("qcdSingleJetFilterMed"))

