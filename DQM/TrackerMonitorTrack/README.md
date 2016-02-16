### Purpose

This code generates the `HitResiduals` and `NormalizedHitResiduals` histograms.

### Data displayed

The data shown in the histograms is computed by the Alignment/OfflineValidation code. It is computed for SiPixel and SiStrip in x- and y-direction (resXprime, resYprime values), however for the strips only the x-residual is relevant.

### Options

* `MonitorTrackResiduals.Mod_On` setting this to `True` will generate plots for every module. By default, only one plot per layer and wheel/disk is generated.

### Bugs

* The plots for pixel disks appear hidden in the shell and half cylinder folders, even though they cover the full layer/disk.

### Currently generated histograms

* `Pixel/Barrel/HitResidualsX_L1`
* `Pixel/Barrel/HitResidualsX_L2`
* `Pixel/Barrel/HitResidualsX_L3`
* `Pixel/Barrel/HitResidualsY_L1`
* `Pixel/Barrel/HitResidualsY_L2`
* `Pixel/Barrel/HitResidualsY_L3`
* `Pixel/Endcap/HitResidualsX_Dm1`
* `Pixel/Endcap/HitResidualsX_Dm2`
* `Pixel/Endcap/HitResidualsX_Dp1`
* `Pixel/Endcap/HitResidualsX_Dp2`
* `Pixel/Endcap/HitResidualsY_Dm1`
* `Pixel/Endcap/HitResidualsY_Dm2`
* `Pixel/Endcap/HitResidualsY_Dp1`
* `Pixel/Endcap/HitResidualsY_Dp2`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_1/HitResiduals_TEC__wheel__1`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_2/HitResiduals_TEC__wheel__2`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_3/HitResiduals_TEC__wheel__3`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_4/HitResiduals_TEC__wheel__4`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_5/HitResiduals_TEC__wheel__5`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_6/HitResiduals_TEC__wheel__6`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_7/HitResiduals_TEC__wheel__7`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_8/HitResiduals_TEC__wheel__8`
* `SiStrip/MechanicalView/TEC/MINUS/wheel_9/HitResiduals_TEC__wheel__9`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_1/HitResiduals_TEC__wheel__1`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_2/HitResiduals_TEC__wheel__2`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_3/HitResiduals_TEC__wheel__3`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_4/HitResiduals_TEC__wheel__4`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_5/HitResiduals_TEC__wheel__5`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_6/HitResiduals_TEC__wheel__6`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_7/HitResiduals_TEC__wheel__7`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_8/HitResiduals_TEC__wheel__8`
* `SiStrip/MechanicalView/TEC/PLUS/wheel_9/HitResiduals_TEC__wheel__9`
* `SiStrip/MechanicalView/TIB/layer_1/HitResiduals_TIB__Layer__1`
* `SiStrip/MechanicalView/TIB/layer_2/HitResiduals_TIB__Layer__2`
* `SiStrip/MechanicalView/TIB/layer_3/HitResiduals_TIB__Layer__3`
* `SiStrip/MechanicalView/TIB/layer_4/HitResiduals_TIB__Layer__4`
* `SiStrip/MechanicalView/TID/PLUS/wheel_1/HitResiduals_TID__wheel__1`
* `SiStrip/MechanicalView/TID/PLUS/wheel_2/HitResiduals_TID__wheel__2`
* `SiStrip/MechanicalView/TID/PLUS/wheel_3/HitResiduals_TID__wheel__3`
* `SiStrip/MechanicalView/TOB/layer_1/HitResiduals_TOB__Layer__1`
* `SiStrip/MechanicalView/TOB/layer_2/HitResiduals_TOB__Layer__2`
* `SiStrip/MechanicalView/TOB/layer_3/HitResiduals_TOB__Layer__3`
* `SiStrip/MechanicalView/TOB/layer_4/HitResiduals_TOB__Layer__4`
* `SiStrip/MechanicalView/TOB/layer_5/HitResiduals_TOB__Layer__5`
* `SiStrip/MechanicalView/TOB/layer_6/HitResiduals_TOB__Layer__6`
