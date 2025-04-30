import FWCore.ParameterSet.Config as cms
                                      
OutALCARECOPromptCalibProdSiPixelAliHLTHGC_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelAliHLTHGMinBias','pathALCARECOPromptCalibProdSiPixelAliHLTHGDiMu')
    ),
    outputCommands = cms.untracked.vstring('keep *_SiPixelAliMillePedeFileConverterHLTHGDimuon_*_*',
                                           'keep *_SiPixelAliMillePedeFileConverterHLTHG_*_*')
)

OutALCARECOPromptCalibProdSiPixelAliHLTHGC=OutALCARECOPromptCalibProdSiPixelAliHLTHGC_noDrop.clone()
OutALCARECOPromptCalibProdSiPixelAliHLTHGC.outputCommands.insert(0, "drop *")
-- dummy change --
