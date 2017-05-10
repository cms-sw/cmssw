#define bTagPostValidation for the b-tag DQM validation (efficiency and mistagrate plot)
process.bTagPostValidation = cms.EDAnalyzer("HLTBTagHarvestingAnalyzer",
HLTPathNames = fileini.btag_pathes,
histoName	= fileini.btag_modules_string,
minTag	= cms.double(0.6),
# MC stuff
mcFlavours = cms.PSet(
light = cms.vuint32(1, 2, 3, 21), # udsg
c = cms.vuint32(4),
b = cms.vuint32(5),
g = cms.vuint32(21),
uds = cms.vuint32(1, 2, 3)
)
)
