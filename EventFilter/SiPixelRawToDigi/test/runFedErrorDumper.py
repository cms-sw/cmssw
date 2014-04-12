#
import FWCore.ParameterSet.Config as cms

process = cms.Process("i")

## import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
## # accept if 'path_1' succeeds
## process.hltfilter = hlt.hltHighLevel.clone(
## # Min-Bias
## #    HLTPaths = ['HLT_Physics_v1'],
## #    HLTPaths = ['HLT_L1Tech_BSC_minBias_threshold1_v1'],
## #    HLTPaths = ['HLT_Random_v1'],
## #    HLTPaths = ['HLT_ZeroBias_v*'],
## # old
## #    HLTPaths = ['HLT_L1_BPTX_MinusOnly','HLT_L1_BPTX_PlusOnly'],
## #    HLTPaths = ['HLT_L1Tech_BSC_minBias'],
## #    HLTPaths = ['HLT_L1Tech_BSC_minBias_OR'],
## # Commissioning:
## #    HLTPaths = ['HLT_L1_Interbunch_BSC_v1'],
## #    HLTPaths = ['HLT_L1_PreCollisions_v1'],
## #    HLTPaths = ['HLT_BeamGas_BSC_v1'],
## #    HLTPaths = ['HLT_BeamGas_HF_v1'],
## # old
## #    HLTPaths = ['HLT_L1_BPTX','HLT_ZeroBias','HLT_L1_BPTX_MinusOnly','HLT_L1_BPTX_PlusOnly'],
## #    HLTPaths = ['p*'],
## #    HLTPaths = ['path_?'],
##     andOr = True,  # False = and, True=or
##     throw = False
##     )

process.MessageLogger = cms.Service("MessageLogger",
     debugModules = cms.untracked.vstring('dumper'),
     destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
     cout = cms.untracked.PSet(
#         threshold = cms.untracked.string('DEBUG')
         threshold = cms.untracked.string('WARNING')
     )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/d/dkotlins/public/test.root'))
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/d/dkotlins/public/digis.root'))
#    fileNames = cms.untracked.vstring('file:/tmp/dkotlins/digis.root'))

#process.out = cms.OutputModule("PoolOutputModule",
#    fileName =  cms.untracked.string('file:histos.root')
#)

process.dumper = cms.EDAnalyzer("FedErrorDumper", 
#    Timing = cms.untracked.bool(False),
    Verbosity = cms.untracked.bool(True),
#    IncludeErrors = cms.untracked.bool(True),
#    InputLabel = cms.untracked.string('source'),
    InputLabel = cms.untracked.string('siPixelDigis'),
#    CheckPixelOrder = cms.untracked.bool(False)
)

# process.p = cms.Path(process.hltfilter*process.dumper)
process.p = cms.Path(process.dumper)

# process.ep = cms.EndPath(process.out)


