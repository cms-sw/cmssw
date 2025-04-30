import FWCore.ParameterSet.Config as cms
                                      
OutALCARECOPromptCalibProdSiPixelAliHGComb_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelAliHGMinBias','pathALCARECOPromptCalibProdSiPixelAliHGDiMu')
    ),
    outputCommands = cms.untracked.vstring('keep *_SiPixelAliMillePedeFileConverterHGDimuon_*_*',
                                           'keep *_SiPixelAliMillePedeFileConverterHG_*_*')
)

OutALCARECOPromptCalibProdSiPixelAliHGComb=OutALCARECOPromptCalibProdSiPixelAliHGComb_noDrop.clone()
OutALCARECOPromptCalibProdSiPixelAliHGComb.outputCommands.insert(0, "drop *")
-- dummy change --
