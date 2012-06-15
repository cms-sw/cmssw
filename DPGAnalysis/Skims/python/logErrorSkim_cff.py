from FWCore.Modules.logErrorFilter_cfi import *
from Configuration.StandardSequences.RawToDigi_Data_cff import gtEvmDigis
stableBeam = cms.EDFilter("HLTBeamModeFilter",
                          L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                          AllowedBeamMode = cms.vuint32(11),
                          saveTags = cms.bool(False)
                          )

logerrorseq=cms.Sequence(gtEvmDigis+stableBeam+logErrorFilter)
