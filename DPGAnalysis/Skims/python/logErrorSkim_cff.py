from FWCore.Modules.logErrorFilter_cfi import *
from Configuration.StandardSequences.RawToDigi_Data_cff import gtEvmDigis
stableBeam = cms.EDFilter("HLTBeamModeFilter",
                          L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                          AllowedBeamMode = cms.vuint32(11)
                          )

logerrorseq=cms.Sequence(gtEvmDigis+stableBeam+logErrorFilter)
