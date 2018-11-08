import FWCore.ParameterSet.Config as cms
from pfDeepDoubleBvLJetTags_cfi import pfDeepDoubleBvLJetTags

pfDeepDoubleCvLJetTags = pfDeepDoubleBvLJetTags.clone(
    flavor = 'CvL',
    graph_path = cms.FileInPath('RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDC.pb'),
    flav_table = cms.PSet(
      probQCD = cms.vuint32(0),
      probHcc = cms.vuint32(1)
    )
)
