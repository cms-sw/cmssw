# Important note:
# due to the limitations of the DBS database schema, as described in
# https://cms-talk.web.cern.ch/t/alcaprompt-datasets-not-loaded-in-dbs/11146/2,
# the keys of the dict (i.e. the "PromptCalib*") MUST be shorter than 31 characters
autoPCL = {'PromptCalibProd'                : 'BeamSpotByRun+BeamSpotByLumi',
           'PromptCalibProdBeamSpotHP'      : 'BeamSpotHPByRun+BeamSpotHPByLumi',
           'PromptCalibProdBeamSpotHPLowPU' : 'BeamSpotHPLowPUByRun+BeamSpotHPLowPUByLumi',
           'PromptCalibProdSiStrip'         : 'SiStripQuality',
           'PromptCalibProdSiStripGains'    : 'SiStripGains',
           'PromptCalibProdSiStripGainsAAG' : 'SiStripGainsAAG',
           'PromptCalibProdSiStripHitEff'   : 'SiStripHitEff',
           'PromptCalibProdSiStripLA'       : 'SiStripLA',
           'PromptCalibProdSiPixelAli'      : 'SiPixelAli',
           'PromptCalibProdSiPixelAliHG'    : 'SiPixelAliHG',
           'PromptCalibProdSiPixelAliHGComb': 'SiPixelAliHGCombined',
           'PromptCalibProdSiPixel'         : 'SiPixelQuality',
           'PromptCalibProdSiPixelLA'       : 'SiPixelLA',
           'PromptCalibProdSiPixelLAMCS'    : 'SiPixelLAMCS',
           'PromptCalibProdEcalPedestals'   : 'EcalPedestals',
           'PromptCalibProdLumiPCC'         : 'LumiPCC',
           'PromptCalibProdPPSTimingCalib'  : 'PPSTimingCalibration',
           'PromptCalibProdPPSDiamondSampic': 'PPSDiamondSampicTimingCalibration',
           'PromptCalibProdPPSAlignment'    : 'PPSAlignment'
           }
