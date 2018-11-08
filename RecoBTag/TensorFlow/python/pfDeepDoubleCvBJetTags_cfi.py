import FWCore.ParameterSet.Config as cms
from pfDeepDoubleBvLJetTags_cfi import pfDeepDoubleBvLJetTags

pfDeepDoubleCvBJetTags = pfDeepDoubleBvLJetTags.clone(
    flavor = 'CvB',
    graph_path = cms.FileInPath('RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDCvB.pb'),
    flav_table = cms.PSet(
      probHbb = cms.vuint32(0),
      probHcc = cms.vuint32(1)
    )
)

