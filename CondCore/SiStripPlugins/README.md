This package contains a series of `cmssw` plugins for inspecting SiStrip conditions.
The available inspectors are:

| Record                          | Object                      | Inspector                                 |
| --------------------------------|-----------------------------| ------------------------------------------|
| `SiStripPedestalsRcd`           | `SiStripPedestals`          | `SiStripPedestals_PayloadInspector.cc`    |    
| `SiStripNoisesRcd`              | `SiStripNoises`      	| `SiStripNoises_PayloadInspector.cc`       |      
| `SiStripLorentzAngleRcd`        | `SiStripLorentzAngle`	| `SiStripLorentzAngle_PayloadInspector.cc` |
| `SiStripLorentzAngleSimRcd`     | `SiStripLorentzAngle`       | `SiStripLorentzAngle_PayloadInspector.cc` |  
| `SiStripBackPlaneCorrectionRcd` | `SiStripBackPlaneCorrection`| `SiStripBackPlaneCorrection_PayloadInspector.cc` |`
| `SiStripApvGainRcd`             | `SiStripApvGain`            | `SiStripApvGain_PayloadInspector.cc`      |  
| `SiStripApvGain2Rcd`            | `SiStripApvGain` 	        | `SiStripApvGain_PayloadInspector.cc`      |
| `SiStripApvGain3Rcd`            | `SiStripApvGain` 	        | `SiStripApvGain_PayloadInspector.cc`      |
| `SiStripApvGainSimRcd`          | `SiStripApvGain` 	        | `SiStripApvGain_PayloadInspector.cc`      |
| `SiStripBadStripRcd`            | `SiStripBadStrip`	        | `SiStripBadStrip_PayloadInspector.cc`     |
| `SiStripBadModuleRcd`           | `SiStripBadStrip`	        | `SiStripBadStrip_PayloadInspector.cc`     |
| `SiStripBadFiberRcd`            | `SiStripBadStrip`	        | `SiStripBadStrip_PayloadInspector.cc`     |
| `SiStripBadChannelRcd`          | `SiStripBadStrip`	        | `SiStripBadStrip_PayloadInspector.cc`     |
| `SiStripDCSStatusRcd`           | `SiStripBadStrip`	        | `SiStripBadStrip_PayloadInspector.cc`     |
| `SiStripDetVOffRcd`             | `SiStripDetVOff` 	        | `SiStripDetVOff_PayloadInspector.cc`      |
| `SiStripConfObjectRcd`          | `SiStripConfObject` 	| `SiStripConfObject_PayloadInspector.cc`   |
| `SiStripThresholdRcd `          | `SiStripThreshold`          | `SiStripThreshold_PayloadInspector.cc`    |
| `SiStripLatencyRcd`             | `SiStripLatency`            | `SiStripLatency_PayloadInspector.cc`      |


Plots will be shown within the **cmsDbBrowser** [payload inspector](https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod) application.
In the `CondCore/SiStripPlugins/test` directory a few bash scripts to inspect conditions from command line are available.
