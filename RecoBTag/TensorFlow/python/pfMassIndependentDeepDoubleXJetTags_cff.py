import FWCore.ParameterSet.Config as cms
from pfDeepDoubleBvLJetTags_cfi import pfDeepDoubleBvLJetTags
from pfDeepDoubleCvLJetTags_cfi import pfDeepDoubleCvLJetTags
from pfDeepDoubleCvBJetTags_cfi import pfDeepDoubleCvBJetTags

pfMassIndependentDeepDoubleBvLJetTags = pfDeepDoubleBvLJetTags.clone(
    graph_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDB_mass_independent.pb')
pfMassIndependentDeepDoubleCvLJetTags = pfDeepDoubleCvLJetTags.clone(
    graph_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDC_mass_independent.pb')
pfMassIndependentDeepDoubleCvBJetTags = pfDeepDoubleCvBJetTags.clone(
    graph_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDCvB_mass_independent.pb')


