import FWCore.ParameterSet.Config as cms
from pfDeepDoubleBvLJetTags_cfi import pfDeepDoubleBvLJetTags

pfMassIndependentDeepDoubleCvLJetTags = pfDeepDoubleBvLJetTags.clone(
    flavor = 'CvL',
    graph_path = cms.FileInPath('RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDC_mass_independent.pb'),
    flav_table = cms.PSet(
      probQCD = cms.vuint32(0),
      probHcc = cms.vuint32(1)
    )
)
