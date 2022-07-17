# Important note:
# due to the limitations of the DBS database schema, as described in
# https://cms-talk.web.cern.ch/t/alcaprompt-datasets-not-loaded-in-dbs/11146/2,
# the keys of the dict (i.e. the "PromptCalib*") MUST be shorter than 31 characters
autoPCL = {'PromptCalibProd' : 'BeamSpotByRun+BeamSpotByLumi',
           'PromptCalibProdBeamSpotHP' : 'BeamSpotHPByRun+BeamSpotHPByLumi',
           'PromptCalibProdBeamSpotHPLowPU' : 'BeamSpotHPLowPUByRun+BeamSpotHPLowPUByLumi',
           'PromptCalibProdSiStrip' : 'SiStripQuality',
           'PromptCalibProdSiStripGains' : 'SiStripGains',
           'PromptCalibProdSiStripGainsAAG' : 'SiStripGainsAAG',
           'PromptCalibProdSiStripHitEff' : 'SiStripHitEff',
           'PromptCalibProdSiPixelAli' : 'SiPixelAli',
           'PromptCalibProdSiPixelAliHG' : 'SiPixelAliHG',
           'PromptCalibProdSiPixel' : 'SiPixelQuality',
           'PromptCalibProdSiPixelLA' : 'SiPixelLA',
           'PromptCalibProdEcalPedestals': 'EcalPedestals',
           'PromptCalibProdLumiPCC': 'LumiPCC',
           'PromptCalibProdPPSTimingCalib' : 'PPSTimingCalibration',
<<<<<<< HEAD
           'PromptCalibProdPPSDiamondSampicTimingCalib' : 'PPSDiamondSampicTimingCalibration',
=======
           'PromptCalibProdPPSDiamondSampic' : 'PPSDiamondSampicTimingCalibration',
>>>>>>> 2b294546c3ee51493450581eb7729a1e5e139fa3
           'PromptCalibProdPPSAlignment' : 'PPSAlignment'
           }
