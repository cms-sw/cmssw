# CondCore/SiPixelPlugins

This package contains a series of `cmssw` plugins for inspecting SiPixel conditions.
The available inspectors are:

| Record                                | Object                          | Inspector                                         |
| --------------------------------------|---------------------------------| --------------------------------------------------|
| `SiPixelLorentzAngleRcd`              | `SiPixelLorentzAngle`           | [SiPixelLorentzAngle_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelLorentzAngle_PayloadInspector.cc) |
| `SiPixelGenErrorDBObjectRcd`          | `SiPixelGenErrorDBObject`       | [SiPixelGenErrorDBObject_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelGenErrorDBObject_PayloadInspector.cc)       |
| `SiPixelGainCalibrationOfflineRcd`    | `SiPixelGainCalibrationOffline` | [SiPixelGainCalibrationOffline_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelGainCalibrationOffline_PayloadInspector.cc) |
| `SiPixelGainCalibrationForHLTRcd`     | `SiPixelGainCalibrationForHLT`  | [SiPixelGainCalibrationForHLT_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelGainCalibrationForHLT_PayloadInspector.cc)  |
| NOT IMPLEMENTED                       | `SiPixelVCal`                   | [SiPixelVCal_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelVCal_PayloadInspector.cc)                   |
| `SiPixelTemplateDBObjectRcd`          | `SiPixelTemplateDBObject`       | [SiPixelTemplateDBObject_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelTemplateDBObject_PayloadInspector.cc)       |
| `SiPixel2DTemplateDBObjectRcd`        | `SiPixel2DTemplateDBObject`     | [SiPixel2DTemplateDBObject_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixel2DTemplateDBObject_PayloadInspector.cc)       |
| `SiPixelQualityFromDbRcd`             | `SiPixelQuality`                | [SiPixelQuality_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelQuality_PayloadInspector.cc)                |
| `SiPixelStatusScenarioProbabilityRcd` | `SiPixelQualityProbabilities`   | [SiPixelQualityProbabilities_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelQualityProbabilities_PayloadInspector.cc)   |
| `SiPixelStatusScenariosRcd`           | `SiPixelFEDChannelContainer`    | [SiPixelFEDChannelContainer_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelFEDChannelContainer_PayloadInspector.cc)    |
| `SiPixelDynamicInefficiencyRcd`       | `SiPixelDynamicInefficiency`    | [SiPixelDynamicInefficiency_PayloadInspector.cc](https://github.com/cms-sw/cmssw/blob/master/CondCore/SiPixelPlugins/plugins/SiPixelDynamicInefficiency_PayloadInspector.cc)  |

Plots will be shown within the **cmsDbBrowser** [payload inspector](https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod) application.
In the `CondCore/SiPixelPlugins/test` directory a few bash scripts to inspect conditions from command line are available.