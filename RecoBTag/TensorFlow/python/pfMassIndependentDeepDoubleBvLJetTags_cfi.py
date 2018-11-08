import FWCore.ParameterSet.Config as cms
from pfDeepDoubleBvLJetTags_cfi import pfDeepDoubleBvLJetTags

pfMassIndependentDeepDoubleBvLJetTags = pfDeepDoubleBvLJetTags.clone(
    flavor = 'BvL',
    graph_path = cms.FileInPath('RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDB_mass_independent.pb'),
    flav_table = cms.PSet(
      probQCD = cms.vuint32(0),
      probHbb = cms.vuint32(1)
    )
)

