import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import FEVTEventContent
skimContent = FEVTEventContent.clone()
skimContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimContent.outputCommands.append("drop *_*_*_SKIM")


from Configuration.Skimming.pwdgSkimBPark_cfi import *
SkimBParkPath = cms.Path(SkimBPark)
SKIMStreamSkimBPark = cms.FilteredStream(
    responsible = 'BPH PAG',
    name = 'SkimBPark',
    paths = ( SkimBParkPath ),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('RAW-RECO')
)

